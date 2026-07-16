"""
細胞セグメンテーション U-Net 学習パイプライン (Fluo-N3DH-SIM+, seq01)

やること（PROJECT.md参照）:
  1. フレーム単位でtrain/val/testに分割し、各フレームから細胞が写っているZ範囲のスライスを使う
  2. 学習データからパーセンタイル(0.5-99.5%)を計算し、クリップ+min-maxで正規化
  3. U-Netを学習（BCE+Dice損失、Adam、Early Stopping）
  4. テストデータでIoUを評価し、予測マスクを1枚可視化保存する

まず動かすことを優先したベースライン実装。
"""

import argparse
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 環境変数DATASET_NAMEで一時的に上書きできる(例: DATASET_NAME=BF-C2DL-HSC python3 main.py ...)。
# 無指定なら従来通りFluo-N3DH-SIM+がデフォルト(COMMANDS.md記載の手順と一致させるため)。
DATASET_NAME = os.environ.get("DATASET_NAME", "Fluo-N3DH-SIM+")

# "binary"(背景/細胞の2値、従来通り) or "boundary3class"(背景/内部/境界の3クラス、新方式)。
# 既存の2値分類の学習結果を再現できるよう、デフォルトは変更しない(PROJECT_labeling.md参照)。
LABEL_MODE = os.environ.get("LABEL_MODE", "binary")
DATA_DIR = Path(__file__).parent.parent / "DATA_CELL" / DATASET_NAME
SEQ = "01"

# データセットごとの構造差分をここに集約する。
#   is_3d: 1フレーム=Zスタック(3Dボリューム)かどうか(BF-C2DL-HSCのみ2D+time、Zなし)
#   frame_digits: ファイル名の連番の桁数(t000.tif vs t0000.tif)
#   annotation_subdir: マスクを"GT"(人手curated)と"ST"(Silver Truth、アルゴリズム生成)の
#     どちらから取るか。Fluo-C3DL-MDA231/Fluo-N3DH-CHOは`GT/SEG`が疎(1フレームにつき
#     数枚のZスライスだけ、Zスタック全体ではない)なので、全フレーム・全Z密な`ST/SEG`を使う。
DATASET_CONFIGS = {
    "Fluo-N3DH-SIM+":   dict(is_3d=True,  frame_digits=3, annotation_subdir="GT"),
    "BF-C2DL-HSC":      dict(is_3d=False, frame_digits=4, annotation_subdir="GT"),
    "Fluo-C3DL-MDA231": dict(is_3d=True,  frame_digits=3, annotation_subdir="ST"),
    "Fluo-N3DH-CHO":    dict(is_3d=True,  frame_digits=3, annotation_subdir="ST"),
}
_cfg = DATASET_CONFIGS[DATASET_NAME]
IS_3D = _cfg["is_3d"]
FRAME_DIGITS = _cfg["frame_digits"]
ANNOTATION_SUBDIR = _cfg["annotation_subdir"]

RESULTS_DIR = Path(__file__).parent / "results" / DATASET_NAME
# MODELS_DIRはLABEL_MODEでネストを分ける。binaryは従来のパス(models/<DATASET_NAME>/)のまま
# 完全後方互換にし、それ以外(boundary3class等)は1段深くして、出力チャンネル数が違う
# チェックポイントを誤ってauto-resumeで読み込んでしまう事故を防ぐ。
MODELS_DIR = Path(__file__).parent / "models" / DATASET_NAME
if LABEL_MODE != "binary":
    MODELS_DIR = MODELS_DIR / LABEL_MODE
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
PAD_TO = 8  # 3回のMaxPool(2x2)で割り切れるように入力をパディングする単位

PERCENTILE_LO, PERCENTILE_HI = 0.5, 99.5
N_FRAMES_FOR_NORM_STATS = 30  # 正規化の統計を計算するのに使うtrainフレーム数

BOUNDARY_WIDTH_PX = 2  # 各インスタンスをこの幅だけ収縮させた残りを「内部」、削れたリングを「境界」とする

# distance_transformモード用の定数(PROJECT_labeling.md参照)。
# WATERSHED_MIN_PEAK_DISTANCE: 距離地形のピーク(細胞の種)を探す時の、ピーク間の最小距離(px)。
#   小さすぎると1つの細胞内に複数の種ができてしまう。
# 前景/背景の判定は(全体を0付近に偏らせる回帰1本ではなく)専用の2値分類チャンネルで
# 行うので、距離側に独立の閾値は不要(2チャンネル方式、詳細はPROJECT_labeling.md参照)。
#
# 深さチャンネルの正規化は固定px数で割る方式(旧DISTANCE_NORM_PX=30)ではなく、
# インスタンスごとに自身の最大距離で割る方式にしている。実測(CHO全フレーム・
# 全インスタンス4188個)で最大距離は1.0〜43.8px(平均24.6px)まで広く分布し、
# 固定30pxで割ると全インスタンスの32.0%(全前景ピクセルの1.2%)がclip(0,1)で
# 同じ値1.0に潰れてしまっていた。インスタンス自身の最大値で割れば、常に
# そのインスタンスの最も深い点が正確に1.0になり、clipが不要(=情報が潰れない)。
WATERSHED_MIN_PEAK_DISTANCE = 10


def raw_path(frame_idx):
    return DATA_DIR / SEQ / f"t{frame_idx:0{FRAME_DIGITS}d}.tif"


def mask_path(frame_idx):
    return DATA_DIR / f"{SEQ}_{ANNOTATION_SUBDIR}" / "SEG" / f"man_seg{frame_idx:0{FRAME_DIGITS}d}.tif"


def list_available_frames():
    """
    マスクが実際に存在するフレーム番号の一覧(昇順)を返す。
    IS_3D=Trueのデータセットは(GTかSTかに関わらず)全フレームに存在する前提なので、
    raw画像ディレクトリの`t*.tif`の実ファイル数で決める(N_FRAMESのハードコードをやめて
    ディスクから自動検出することで、データセットごとの総フレーム数の違いにも対応する)。
    BF-C2DL-HSCは疎なアノテーション(1764フレーム中49枚のみ)なので、
    実際にman_seg*.tifが存在するファイルをディスクから探して決める。
    """
    if IS_3D:
        return sorted(int(p.stem[1:]) for p in (DATA_DIR / SEQ).glob("t*.tif"))
    seg_dir = DATA_DIR / f"{SEQ}_{ANNOTATION_SUBDIR}" / "SEG"
    prefix_len = len("man_seg")
    return sorted(int(p.stem[prefix_len:]) for p in seg_dir.glob("man_seg*.tif"))


def derive_class_labels_2d(instance_slice, boundary_width=BOUNDARY_WIDTH_PX):
    """
    2DのインスタンスIDラベル画像(0=背景、1,2,3,...=各細胞の固有ID)から、
    {0:背景, 1:内部, 2:境界}のクラスラベルを導出する(PROJECT_labeling.md参照)。

    各インスタンスを個別に収縮させることが重要: 結合した2値マスク全体をまとめて
    収縮すると、接触している2つのインスタンスは1つの塊のまま収縮されるだけで
    分離されない。インスタンスごとに収縮してから合わせることで、接触部分に
    「境界」クラスのリングが自然に生まれる。
    """
    binary = instance_slice > 0
    interior = np.zeros_like(binary, dtype=bool)
    for inst_id in np.unique(instance_slice):
        if inst_id == 0:
            continue
        inst_binary = instance_slice == inst_id
        interior |= ndimage.binary_erosion(inst_binary, iterations=boundary_width)
    labels = np.zeros(instance_slice.shape, dtype=np.int64)
    labels[binary] = 2  # 境界(後で内部を上書きする)
    labels[interior] = 1  # 内部
    return labels


def derive_distance_target_2d(instance_slice):
    """
    2DのインスタンスIDラベル画像(0=背景、1,2,3,...=各細胞の固有ID)から、
    各前景ピクセルについて「そのインスタンス自身の外(=背景、または隣接する
    別インスタンス)までの最短距離」を計算する(PROJECT_labeling.md参照)。

    derive_class_labels_2dと同じ理由で、インスタンスごとに個別に
    distance_transform_edtを適用することが重要: 結合した2値マスク全体に
    1回だけ適用すると、接触している隣のインスタンスも「同じ前景」と見なされて
    しまい、その方向には距離が縮まらない(=境界での分離が失われる)。
    """
    distances = np.zeros(instance_slice.shape, dtype=np.float32)
    for inst_id in np.unique(instance_slice):
        if inst_id == 0:
            continue
        inst_binary = instance_slice == inst_id
        dt = ndimage.distance_transform_edt(inst_binary)
        distances[inst_binary] = dt[inst_binary]
    return distances


def build_samples_with_mask_cache(frame_list):
    """
    各フレームのGTマスク全体を1回だけ読み込み、学習サンプルのリストとマスクの
    キャッシュ(dict)を同時に作る。学習中に同じマスクをディスクから読み直さずに済むため。
    インスタンスIDつきの生のマスクをそのままキャッシュする(2値化はCellSliceDataset側で
    LABEL_MODEに応じて行う。boundary3classでは境界導出にIDが必要なため)。

    3Dデータセット(IS_3D=True)は、細胞が写っているzスライスごとに(frame_idx, z)を
    1サンプルとする。2Dデータセットは1フレームがそのまま1サンプルなので(frame_idx, None)。
    """
    samples = []
    mask_cache = {}
    for f in frame_list:
        mask = tifffile.imread(mask_path(f))
        binary_mask = (mask > 0)
        if IS_3D:
            nonzero = np.where(binary_mask.sum(axis=(1, 2)) > 0)[0]
            if len(nonzero) == 0:
                continue
            mask_cache[f] = mask
            for z in range(int(nonzero.min()), int(nonzero.max()) + 1):
                samples.append((f, z))
        else:
            if binary_mask.sum() == 0:
                continue
            mask_cache[f] = mask
            samples.append((f, None))
    return samples, mask_cache


def build_frame_split(seed=SEED):
    frames = list_available_frames()
    rng = random.Random(seed)
    rng.shuffle(frames)
    n_train = int(len(frames) * 0.7)
    n_val = int(len(frames) * 0.2)
    train_frames = sorted(frames[:n_train])
    val_frames = sorted(frames[n_train:n_train + n_val])
    test_frames = sorted(frames[n_train + n_val:])
    return train_frames, val_frames, test_frames


def compute_norm_bounds(train_frames, seed=SEED):
    """train frameのみからパーセンタイル境界値を計算する（val/testはこの値を使い回す）"""
    rng = random.Random(seed)
    sample_frames = rng.sample(train_frames, min(N_FRAMES_FOR_NORM_STATS, len(train_frames)))
    pixels = [tifffile.imread(raw_path(f)).ravel() for f in sample_frames]
    all_pixels = np.concatenate(pixels)
    lo, hi = np.percentile(all_pixels, [PERCENTILE_LO, PERCENTILE_HI])
    return float(lo), float(hi)


def compute_abs_bounds(train_frames, seed=SEED):
    """train frameのみから絶対min-max(クリップ無し)を計算する。可視化の「input」パネル用。"""
    rng = random.Random(seed)
    sample_frames = rng.sample(train_frames, min(N_FRAMES_FOR_NORM_STATS, len(train_frames)))
    pixels = [tifffile.imread(raw_path(f)).ravel() for f in sample_frames]
    all_pixels = np.concatenate(pixels)
    return float(all_pixels.min()), float(all_pixels.max())


class CellSliceDataset(Dataset):
    def __init__(self, samples, mask_cache, norm_lo, norm_hi):
        self.samples = samples
        self.mask_cache = mask_cache
        self.norm_lo = norm_lo
        self.norm_hi = norm_hi

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_idx, z = self.samples[idx]
        # maskは事前にbuild_samples_with_mask_cacheでキャッシュ済みのものを使う（再読み込みしない）
        if z is None:
            # 2Dデータセット: ファイル自体が1枚の画像なのでkey指定は不要
            raw = tifffile.imread(raw_path(frame_idx)).astype(np.float32)
            instance_slice = self.mask_cache[frame_idx]
        else:
            # 3Dデータセット: key=zでそのページだけ読む（ボリューム全体を読み込まない）
            raw = tifffile.imread(raw_path(frame_idx), key=z).astype(np.float32)
            instance_slice = self.mask_cache[frame_idx][z]

        image = np.clip((raw - self.norm_lo) / (self.norm_hi - self.norm_lo), 0, 1)
        image_t = torch.from_numpy(image).unsqueeze(0)

        return image_t, LABEL_SPEC.build_target(instance_slice)


# ---------------------------------------------------------------- U-Net ----

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, out_channels=1, activation="sigmoid", in_channels=1):
        super().__init__()
        self.activation = activation
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out_conv(d1)
        if self.activation == "sigmoid":
            return torch.sigmoid(out)
        if self.activation == "tanh":
            # dwt.pyのDirection Network専用: 全チャンネルが方向ベクトル成分([-1,1])。
            return torch.tanh(out)
        if self.activation == "sigmoid_tanh2":
            # 末尾2チャンネル(方向ベクトルdy, dx)はtanh([-1,1])、それ以外
            # (前景・深さ)はsigmoid([0,1])。distance_flow_trainedモード専用
            # (PROJECT_labeling2.md参照)。
            n = out.shape[1]
            head = torch.sigmoid(out[:, :n - 2])
            tail = torch.tanh(out[:, n - 2:])
            return torch.cat([head, tail], dim=1)
        # "none": 多クラス(softmax+cross_entropy用)は生のlogitsを返す。
        # F.cross_entropyが内部でlog-softmaxを計算するため、ここでsoftmaxを
        # 適用すると数値的に不安定になる。
        return out


def pad_to_multiple(x, multiple=PAD_TO):
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (h, w)


def dice_loss(pred, target, eps=1e-6):
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return 1 - ((2 * intersection + eps) / (union + eps)).mean()


def bce_dice_loss(pred, target):
    return F.binary_cross_entropy(pred, target) + dice_loss(pred, target)


def multiclass_dice_loss(pred_probs, target_onehot, eps=1e-6):
    """pred_probs, target_onehot: (B, C, H, W)。クラスごとのDice係数の平均を1から引く。"""
    pred_flat = pred_probs.reshape(pred_probs.size(0), pred_probs.size(1), -1)
    target_flat = target_onehot.reshape(target_onehot.size(0), target_onehot.size(1), -1)
    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
    dice_per_class = (2 * intersection + eps) / (union + eps)
    return 1 - dice_per_class.mean()


def ce_dice_loss(logits, target_labels):
    """logits: (B,C,H,W)の生スコア。target_labels: (B,H,W)のLongTensor(クラスID)。
    bce_dice_lossの多クラス版(BCE→categorical cross-entropyに一般化、詳細はPROJECT_labeling.md)。"""
    ce = F.cross_entropy(logits, target_labels)
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target_labels, num_classes=logits.size(1)).permute(0, 3, 1, 2).float()
    return ce + multiclass_dice_loss(probs, target_onehot)


def iou_score(pred_binary, target, eps=1e-6):
    pred_binary = pred_binary.reshape(pred_binary.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = (pred_binary * target).sum(dim=1)
    union = ((pred_binary + target) > 0).float().sum(dim=1)
    return ((intersection + eps) / (union + eps)).mean().item()


def iou_score_multiclass(logits, target_labels, eps=1e-6):
    """背景以外(前景=クラスID!=0)としてのIoU。2値版iou_scoreと同じ定義なので、
    binaryモードの実験結果(test mean IoU)と直接比較できる。"""
    pred_fg = (logits.argmax(dim=1) != 0).float().reshape(logits.size(0), -1)
    target_fg = (target_labels != 0).float().reshape(target_labels.size(0), -1)
    intersection = (pred_fg * target_fg).sum(dim=1)
    union = ((pred_fg + target_fg) > 0).float().sum(dim=1)
    return ((intersection + eps) / (union + eps)).mean().item()


# findContoursで輪郭を抽出する共通ヘルパー。元track_and_visualize.pyにあったが、
# 各LABEL_MODEのto_contour_input(下記)が3モード共通で使うため、循環importを避けて
# こちらに置く(挙動は変えず単純に移動しただけ)。
MIN_CONTOUR_AREA = 5


def get_contours(binary_mask):
    contours, _ = cv2.findContours(binary_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]


# ------------------------------------------------------ LABEL_MODE registry ----
# LABEL_MODEごとに分岐していた処理(学習ターゲットの作り方・損失・IoU・輪郭抽出・
# 可視化)を1箇所にまとめる。既存の個別関数(上記のbce_dice_loss等)は名前も内容も
# 変えず、ここでは「束ねて参照する」だけ(詳細はPROJECT_labeling.md参照)。
# 新しいモードを追加する時は、ここに1エントリ追加するだけでよい。
#
# to_contour_input(pred) は (輪郭リスト, 表示用float map) を返す。watershedのように
# 複数の分離済み領域を扱うモードがあるため、「1枚の2値マスク」ではなく
# 「輪郭リスト」を共通の返り値の形にしている(1枚の2値マスクに戻すと、接触した
# インスタンス同士が再結合してしまうため)。

@dataclass
class LabelModeSpec:
    out_channels: int
    activation: str  # UNetの最終層の活性化("sigmoid" or "none"、詳細はUNet.forward参照)
    build_target: Callable[[np.ndarray], torch.Tensor]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    iou_fn: Callable[[torch.Tensor, torch.Tensor], float]
    to_contour_input: Callable[[torch.Tensor], tuple]
    visualize: Callable[[torch.Tensor, torch.Tensor], dict]
    # track_and_visualize.pyの「output」パネル(1枚のfloat mapをそのまま表示するだけの
    # パネル)の見せ方。to_contour_inputの2番目の戻り値(表示用float map)をどの範囲・
    # どのカラーマップで解釈するかをここで指定する(表示のみ、学習・追跡には無関係)。
    output_cmap: str = "gray"
    output_vmax: float = 1.0
    output_title: str = "output (B:bg, W:cell)"


def _binary_build_target(instance_slice):
    binary_mask = (instance_slice > 0).astype(np.float32)
    return torch.from_numpy(binary_mask).unsqueeze(0)


def _boundary3class_build_target(instance_slice):
    return torch.from_numpy(derive_class_labels_2d(instance_slice)).long()


def _distance_build_target(instance_slice):
    """チャンネル0=前景か否か(0/1、今までのbinary方式と同じ)、
    チャンネル1=内部の深さ(前景のみ意味を持つ、背景・境界は0)。
    2チャンネルに分けることで、深さの回帰を前景ピクセルだけに限定でき、
    背景が大部分を占める偏った分布を深さ側の学習対象から除外できる
    (単一チャンネルで回帰させると勾配消失で学習が停止した。PROJECT_labeling.md参照)。

    深さは固定px数で割るのではなく、インスタンスごとに自身の最大距離で割って
    正規化する(常にそのインスタンスの最も深い点が正確に1.0になり、clipが不要
    =情報が潰れない。詳細はPROJECT_labeling.md参照)。"""
    fg = (instance_slice > 0).astype(np.float32)
    dist = derive_distance_target_2d(instance_slice)
    depth = np.zeros_like(dist, dtype=np.float32)
    for inst_id in np.unique(instance_slice):
        if inst_id == 0:
            continue
        inst_mask = instance_slice == inst_id
        max_d = dist[inst_mask].max()
        if max_d > 0:
            depth[inst_mask] = dist[inst_mask] / max_d
    return torch.from_numpy(np.stack([fg, depth], axis=0))


def _binary_to_contour_input(pred, image=None):
    """pred: (1,1,H,W)のsigmoid済み確率。(輪郭リスト, 表示用確率マップ)を返す。"""
    prob = pred[0, 0].cpu().numpy()
    binary = (prob > 0.5).astype(np.uint8)
    return get_contours(binary), prob


# binary_intensityモード用の定数。前景マスクの範囲内で、元の入力画像の明るさを
# 地形としてwatershedにかけ、接触した細胞を分離する。細胞内部の模様など
# ピクセル単位の輝度ノイズでピークが暴れるのを防ぐため、ピーク探索・watershedの
# 前にガウシアンぼかしをかけて粗くする。
INTENSITY_BLUR_SIGMA = 20.0


def _binary_intensity_to_contour_input(pred, image=None):
    """pred: (1,1,H,W)のsigmoid済み確率(binaryモードと同じ、前景か否か)。
    image: 正規化済み入力画像(H,W)、[0,1]。前景かどうかはモデルの予測(pred)で
    決め、その前景の中だけで、元画像の明るさ(ぼかした後)を地形とみなした
    watershed法で領域を分離する。モデルは前景/背景の判定にしか使わないため、
    binaryモードで既に学習済みのチェックポイントをそのまま再利用できる
    (再学習不要)。
    """
    prob = pred[0, 0].cpu().numpy()
    fg_mask = prob > 0.5

    smoothed = ndimage.gaussian_filter(image, sigma=INTENSITY_BLUR_SIGMA)

    coords = peak_local_max(smoothed, min_distance=WATERSHED_MIN_PEAK_DISTANCE, labels=fg_mask)
    if len(coords) == 0:
        return [], prob

    seed_mask = np.zeros(smoothed.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(seed_mask)

    labels = watershed(-smoothed, markers, mask=fg_mask)
    contours = []
    for label_id in range(1, labels.max() + 1):
        binary = (labels == label_id).astype(np.uint8)
        contours.extend(get_contours(binary))

    return contours, prob


def _boundary3class_to_contour_input(pred, image=None):
    """pred: (1,3,H,W)の生logits。「内部」クラスの2値化を輪郭抽出に使う
    (境界クラスが壁として働くので、内部クラスだけでfindContoursすれば
    接触した細胞が分離される。PROJECT_labeling.md参照)。"""
    probs = torch.softmax(pred, dim=1)[0].cpu().numpy()  # (3, H, W): 背景/内部/境界
    interior_prob = probs[1]
    binary = (probs.argmax(axis=0) == 1).astype(np.uint8)
    return get_contours(binary), interior_prob


def _distance_to_contour_input(pred, image=None):
    """pred: (1,2,H,W)のsigmoid済み確率(チャンネル0=前景, チャンネル1=深さ)。

    前景かどうかは(ノイズに弱い小さな深さの閾値ではなく)従来通り信頼できる
    チャンネル0の2値化(0.5)で決め、その前景の中だけでチャンネル1(深さ)を
    地形とみなしたwatershed法で領域を分離する。ラベルをまとめて2値化してから
    輪郭を取ると隣接する別ラベルの領域が再結合してしまうため、ラベルごとに
    個別処理することが必須(PROJECT_labeling.md参照)。
    """
    fg_prob = pred[0, 0].cpu().numpy()
    depth_map = pred[0, 1].cpu().numpy()
    fg_mask = fg_prob > 0.5

    coords = peak_local_max(depth_map, min_distance=WATERSHED_MIN_PEAK_DISTANCE, labels=fg_mask)
    if len(coords) == 0:
        return [], fg_prob

    seed_mask = np.zeros(depth_map.shape, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(seed_mask)

    labels = watershed(-depth_map, markers, mask=fg_mask)
    contours = []
    for label_id in range(1, labels.max() + 1):
        binary = (labels == label_id).astype(np.uint8)
        contours.extend(get_contours(binary))

    # 表示専用: 背景(a=0)は0(黒)、前景(a=1)は深さb(0=表面付近, 1=そのインスタンス内で
    # 最も深い点、インスタンスごとの相対値なのでpx単位には戻せない)。
    # 輪郭抽出・追跡には使わない、可視化パネル専用の値。
    depth_display = depth_map * fg_mask
    return contours, depth_display


# distance_flowモード用の定数(PROJECT_labeling2.md参照)。
# FLOW_N_STEPS: 軌跡積分の反復回数。実測(CHO全フレーム)でインスタンスの最大距離は
#   43.8pxだったので、1px刻みでも最も遠いピクセルが中心に到達できるよう余裕を持たせた値。
# FLOW_STEP_SIZE: 1回の反復で動く距離(px)。
# FLOW_CLUSTER_DILATION_PX: 積分後の着地点をクラスタリングする際、離散化による
#   ばらつきを吸収するための膨張半径。
FLOW_N_STEPS = 80
FLOW_STEP_SIZE = 1.0
FLOW_CLUSTER_DILATION_PX = 3


def _derive_unit_flow_from_depth(depth_map, fg_mask):
    """予測した深さチャンネル(depth_map)を数値微分し、前景内で長さ1に正規化した
    方向ベクトル場(flow_y, flow_x)を返す(PROJECT_labeling2.md参照)。

    「境界までの最短距離」の勾配は数学的にほぼ至る所で大きさ1になる(実測でも
    確認済み)ため、熱拡散シミュレーションのような別のシミュレーションを行わず、
    既存の深さチャンネルの近傍ピクセル差分(np.gradient)からそのまま方向を導出できる。
    """
    grad_y, grad_x = np.gradient(depth_map)
    mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
    flow_y = np.zeros_like(depth_map)
    flow_x = np.zeros_like(depth_map)
    valid = fg_mask & (mag > 1e-6)
    flow_y[valid] = grad_y[valid] / mag[valid]
    flow_x[valid] = grad_x[valid] / mag[valid]
    return flow_y, flow_x


def _integrate_and_cluster(fg_mask, flow_y, flow_x,
                            n_steps=FLOW_N_STEPS, step_size=FLOW_STEP_SIZE):
    """各前景ピクセルを出発点として、自身のflowベクトルの向きに小さく何度も動かし
    (軌跡積分)、最終的にどこに集まるか(着地点)でグルーピングする
    (Cellpose式のflow場によるインスタンス分離と同じ考え方。PROJECT_labeling2.md参照)。

    戻り値: fg_maskと同じ形の整数ラベル配列(0=背景、1,2,...=各クラスタ)。
    """
    ys, xs = np.where(fg_mask)
    if len(ys) == 0:
        return np.zeros(fg_mask.shape, dtype=np.int64)

    pos_y = ys.astype(np.float64)
    pos_x = xs.astype(np.float64)
    for _ in range(n_steps):
        # 現在位置(サブピクセル)でのflowベクトルを補間して取得し、その向きに進める
        step_y = ndimage.map_coordinates(flow_y, [pos_y, pos_x], order=1, mode="nearest")
        step_x = ndimage.map_coordinates(flow_x, [pos_y, pos_x], order=1, mode="nearest")
        pos_y = np.clip(pos_y + step_size * step_y, 0, fg_mask.shape[0] - 1)
        pos_x = np.clip(pos_x + step_size * step_x, 0, fg_mask.shape[1] - 1)

    landing_y = np.round(pos_y).astype(np.int64)
    landing_x = np.round(pos_x).astype(np.int64)

    landing_mask = np.zeros(fg_mask.shape, dtype=bool)
    landing_mask[landing_y, landing_x] = True
    landing_mask = ndimage.binary_dilation(landing_mask, iterations=FLOW_CLUSTER_DILATION_PX)
    cluster_map, _ = ndimage.label(landing_mask)

    labels = np.zeros(fg_mask.shape, dtype=np.int64)
    labels[ys, xs] = cluster_map[landing_y, landing_x]
    return labels


def _distance_flow_to_contour_input(pred, image=None):
    """pred: (1,2,H,W)のsigmoid済み確率(チャンネル0=前景, チャンネル1=深さ)。

    watershed(深さ地形の山を種にする)の代わりに、深さチャンネルの勾配から
    導いた方向ベクトル場に沿って各前景ピクセルを軌跡積分し、収束先でグルーピングする
    (設計A、PROJECT_labeling2.md参照)。学習済みモデル・build_target・lossは
    distance_transformモードと完全に同じで、後処理のアルゴリズムだけが異なる。
    """
    fg_prob = pred[0, 0].cpu().numpy()
    depth_map = pred[0, 1].cpu().numpy()
    fg_mask = fg_prob > 0.5

    flow_y, flow_x = _derive_unit_flow_from_depth(depth_map, fg_mask)
    labels = _integrate_and_cluster(fg_mask, flow_y, flow_x)

    contours = []
    for label_id in range(1, labels.max() + 1):
        binary = (labels == label_id).astype(np.uint8)
        contours.extend(get_contours(binary))

    depth_display = depth_map * fg_mask
    return contours, depth_display


def _distance_flow_trained_build_target(instance_slice):
    """(前景, 深さ, dy, dx)の4チャンネルを返す。前景・深さは既存の
    `_distance_build_target`と全く同じ。dy, dxは、そのGT(正解データ)の深さから
    `_derive_unit_flow_from_depth`で計算した"綺麗な"(ノイズのない)方向で、
    モデルに直接学習させる教師データにする(PROJECT_labeling2.md参照。
    distance_flowモードのような推論時の後付け微分は行わない)。"""
    fg_depth = _distance_build_target(instance_slice)  # (2, H, W): fg, depth
    fg = fg_depth[0].numpy()
    depth = fg_depth[1].numpy()
    flow_y, flow_x = _derive_unit_flow_from_depth(depth, fg > 0)
    return torch.from_numpy(np.stack([fg, depth, flow_y, flow_x], axis=0))


def _distance_flow_trained_loss(pred, target):
    """pred, target: (B,4,H,W)。前景=bce_dice_loss、深さ・dy・dxは前景マスクした
    MSE(それぞれ独立に、既存の_distance_lossと同じマスクパターン)。"""
    fg_pred, depth_pred = pred[:, 0:1], pred[:, 1:2]
    dy_pred, dx_pred = pred[:, 2:3], pred[:, 3:4]
    fg_target, depth_target = target[:, 0:1], target[:, 1:2]
    dy_target, dx_target = target[:, 2:3], target[:, 3:4]

    fg_loss = bce_dice_loss(fg_pred, fg_target)
    fg_mask = fg_target > 0
    if fg_mask.any():
        depth_loss = F.mse_loss(depth_pred[fg_mask], depth_target[fg_mask])
        dy_loss = F.mse_loss(dy_pred[fg_mask], dy_target[fg_mask])
        dx_loss = F.mse_loss(dx_pred[fg_mask], dx_target[fg_mask])
    else:
        depth_loss = dy_loss = dx_loss = torch.zeros((), device=pred.device)
    return fg_loss + depth_loss + dy_loss + dx_loss


def _distance_flow_trained_to_contour_input(pred, image=None):
    """pred: (1,4,H,W)。チャンネル0=前景(sigmoid)、1=深さ(sigmoid)、
    2,3=モデルが直接予測したdy, dx(tanh)。distance_flowと違い、推論時に
    深さを微分し直す必要がない(モデル自身が滑らかな方向を直接出す前提。
    PROJECT_labeling2.md参照)。"""
    fg_prob = pred[0, 0].cpu().numpy()
    depth_map = pred[0, 1].cpu().numpy()
    flow_y = pred[0, 2].cpu().numpy()
    flow_x = pred[0, 3].cpu().numpy()
    fg_mask = fg_prob > 0.5

    labels = _integrate_and_cluster(fg_mask, flow_y, flow_x)
    contours = []
    for label_id in range(1, labels.max() + 1):
        binary = (labels == label_id).astype(np.uint8)
        contours.extend(get_contours(binary))

    depth_display = depth_map * fg_mask
    return contours, depth_display


# 背景=黒、内部=白、境界=赤(3クラス表示用のカラーマップ)
_CLASS_COLORS = np.array([[0, 0, 0], [255, 255, 255], [255, 60, 60]], dtype=np.uint8)


def _binary_visualize(mask, pred):
    return dict(
        gt_panel=mask.numpy()[0], pred_panel=(pred > 0.5).float().cpu().numpy()[0, 0],
        gt_title="GT mask", pred_title="predicted mask", cmap="gray",
    )


def _boundary3class_visualize(mask, pred):
    return dict(
        gt_panel=_CLASS_COLORS[mask.numpy()], pred_panel=_CLASS_COLORS[pred.argmax(dim=1).cpu().numpy()[0]],
        gt_title="GT (black=bg, white=interior, red=boundary)", pred_title="predicted", cmap=None,
    )


def _distance_visualize(mask, pred):
    return dict(
        gt_panel=mask[1].numpy(), pred_panel=pred[0, 1].cpu().numpy(),
        gt_title="GT depth (foreground only)", pred_title="predicted depth", cmap="viridis",
    )


def _distance_loss(pred, target):
    """pred, target: (B,2,H,W)。チャンネル0(前景)はbce_dice_loss、チャンネル1(深さ)は
    前景ピクセルだけに限定したMSE(PROJECT_labeling.md参照)。"""
    fg_pred, depth_pred = pred[:, 0:1], pred[:, 1:2]
    fg_target, depth_target = target[:, 0:1], target[:, 1:2]
    fg_loss = bce_dice_loss(fg_pred, fg_target)
    fg_mask = fg_target > 0
    if fg_mask.any():
        depth_loss = F.mse_loss(depth_pred[fg_mask], depth_target[fg_mask])
    else:
        depth_loss = torch.zeros((), device=pred.device)
    return fg_loss + depth_loss


def _distance_iou(pred, target):
    return iou_score((pred[:, 0:1] > 0.5).float(), target[:, 0:1])


LABEL_MODES = {
    "binary": LabelModeSpec(
        out_channels=1,
        activation="sigmoid",
        build_target=_binary_build_target,
        loss_fn=bce_dice_loss,
        iou_fn=lambda pred, target: iou_score((pred > 0.5).float(), target),
        to_contour_input=_binary_to_contour_input,
        visualize=_binary_visualize,
    ),
    # binaryモードと学習(build_target/loss/iou/visualize)は完全に同じ、輪郭抽出だけ
    # 「前景マスクをそのままfindContours」から「前景マスクの中で、元の入力画像の
    # 明るさをwatershedの地形として使う」に変えたモード。再学習不要で同じ
    # チェックポイントを読み込める。モデルの中を通した後の特徴ではなく、元画像の
    # 生の輝度値を後付けで直接使うことで、幾何学的な情報(前景の形)だけでは
    # 区別できない接触境界の手がかりを試す。
    "binary_intensity": LabelModeSpec(
        out_channels=1,
        activation="sigmoid",
        build_target=_binary_build_target,
        loss_fn=bce_dice_loss,
        iou_fn=lambda pred, target: iou_score((pred > 0.5).float(), target),
        to_contour_input=_binary_intensity_to_contour_input,
        visualize=_binary_visualize,
    ),
    "boundary3class": LabelModeSpec(
        out_channels=3,
        activation="none",
        build_target=_boundary3class_build_target,
        loss_fn=ce_dice_loss,
        iou_fn=iou_score_multiclass,
        to_contour_input=_boundary3class_to_contour_input,
        visualize=_boundary3class_visualize,
    ),
    "distance_transform": LabelModeSpec(
        out_channels=2,
        activation="sigmoid",
        build_target=_distance_build_target,
        loss_fn=_distance_loss,
        iou_fn=_distance_iou,
        to_contour_input=_distance_to_contour_input,
        visualize=_distance_visualize,
        output_cmap="viridis",
        output_vmax=1.0,
        output_title="output (black=bg, color=depth, relative 0-1 per instance)",
    ),
    # distance_transformと学習(build_target/loss/iou/visualize)は完全に同じ、
    # 推論時の輪郭抽出だけを「watershed」から「深さの勾配に沿った軌跡積分+
    # クラスタリング(設計A)」に変えたモード。再学習不要で同じチェックポイントを
    # 読み込める(PROJECT_labeling2.md参照)。
    "distance_flow": LabelModeSpec(
        out_channels=2,
        activation="sigmoid",
        build_target=_distance_build_target,
        loss_fn=_distance_loss,
        iou_fn=_distance_iou,
        to_contour_input=_distance_flow_to_contour_input,
        visualize=_distance_visualize,
        output_cmap="viridis",
        output_vmax=1.0,
        output_title="output (black=bg, color=depth, relative 0-1 per instance)",
    ),
    # distance_flowと違い、方向(dy, dx)をGTの深さから計算した"綺麗な"値を教師データ
    # として、モデルに直接学習させる(4チャンネル出力)。推論時は深さを微分し直さず、
    # モデルが直接出す方向をそのまま使う(PROJECT_labeling2.md参照)。
    "distance_flow_trained": LabelModeSpec(
        out_channels=4,
        activation="sigmoid_tanh2",
        build_target=_distance_flow_trained_build_target,
        loss_fn=_distance_flow_trained_loss,
        iou_fn=_distance_iou,
        to_contour_input=_distance_flow_trained_to_contour_input,
        visualize=_distance_visualize,
        output_cmap="viridis",
        output_vmax=1.0,
        output_title="output (black=bg, color=depth, relative 0-1 per instance)",
    ),
}
LABEL_SPEC = LABEL_MODES[LABEL_MODE]


def run_epoch(model, loader, optimizer, device, train, target_util=None):
    model.train(train)
    total_loss = 0.0
    n = 0
    for image, mask in loader:
        t0 = time.time()
        image, mask = image.to(device), mask.to(device)
        image_p, (h, w) = pad_to_multiple(image)

        with torch.set_grad_enabled(train):
            pred = model(image_p)[:, :, :h, :w]
            loss = LABEL_SPEC.loss_fn(pred, mask)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # GPU使用率を平均target_utilに抑えるため、計算にかかった時間に応じて小休止を入れる
        if target_util is not None and target_util < 1.0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            busy = time.time() - t0
            time.sleep(busy * (1 / target_util - 1))

        total_loss += loss.item() * image.size(0)
        n += image.size(0)
    return total_loss / n


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, patience_counter):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    }, path)


def load_checkpoint_if_exists(path, model, optimizer, device):
    """既存のチェックポイントがあれば読み込んで(次に始めるepoch, best_val_loss, patience_counter)を返す。
    無ければ(0, inf, 0)を返し、ゼロから学習を始める。"""
    if not path.exists():
        return 0, float("inf"), 0
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"チェックポイントから再開: epoch={ckpt['epoch']} best_val_loss={ckpt['best_val_loss']:.4f}")
    return ckpt["epoch"] + 1, ckpt["best_val_loss"], ckpt["patience_counter"]


def save_sample_prediction(model, dataset, device, path):
    image, mask = dataset[0]
    image_b = image.unsqueeze(0).to(device)
    image_p, (h, w) = pad_to_multiple(image_b)
    with torch.no_grad():
        pred = model(image_p)[:, :, :h, :w]

    panels = LABEL_SPEC.visualize(mask, pred)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.numpy()[0], cmap="gray")
    axes[0].set_title("input")
    axes[1].imshow(panels["gt_panel"], cmap=panels["cmap"])
    axes[1].set_title(panels["gt_title"])
    axes[2].imshow(panels["pred_panel"], cmap=panels["cmap"])
    axes[2].set_title(panels["pred_title"])
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-vram-gb", type=float, default=None,
                         help="このプロセスが使うVRAMの上限(GB)。指定した場合、超えるとOOMエラーになる")
    parser.add_argument("--max-gpu-util", type=float, default=None,
                         help="平均GPU使用率の目標値(0-1)。例えば0.8なら、計算時間に応じて小休止を入れて平均80%程度に抑える")
    parser.add_argument("--quick", action="store_true", help="少数フレーム・少数エポックで動作確認する")
    args = parser.parse_args()
    if args.quick and args.epochs == 200:
        args.epochs = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if device.type == "cuda" and args.max_vram_gb is not None:
        total_mem = torch.cuda.get_device_properties(0).total_memory
        fraction = min((args.max_vram_gb * 1024 ** 3) / total_mem, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        print(f"VRAM上限: {args.max_vram_gb}GB (device全体の{fraction * 100:.1f}%)に設定")

    train_frames, val_frames, test_frames = build_frame_split()
    if args.quick:
        train_frames, val_frames, test_frames = train_frames[:3], val_frames[:2], test_frames[:1]
    print(f"frames: train={len(train_frames)} val={len(val_frames)} test={len(test_frames)}")

    norm_lo, norm_hi = compute_norm_bounds(train_frames)
    print(f"normalization bounds (percentile {PERCENTILE_LO}-{PERCENTILE_HI}%): {norm_lo:.1f} - {norm_hi:.1f}")

    train_samples, train_mask_cache = build_samples_with_mask_cache(train_frames)
    val_samples, val_mask_cache = build_samples_with_mask_cache(val_frames)
    test_samples, test_mask_cache = build_samples_with_mask_cache(test_frames)
    print(f"samples: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

    train_ds = CellSliceDataset(train_samples, train_mask_cache, norm_lo, norm_hi)
    val_ds = CellSliceDataset(val_samples, val_mask_cache, norm_lo, norm_hi)
    test_ds = CellSliceDataset(test_samples, test_mask_cache, norm_lo, norm_hi)

    # 注: このコンテナは/dev/shmが小さく(64MB)、num_workers>0だとワーカーがbus errorで
    # 落ちるため、num_workers=0(メインプロセスで読み込み)にしている
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet(out_channels=LABEL_SPEC.out_channels, activation=LABEL_SPEC.activation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_path = MODELS_DIR / "unet_best.pth"
    start_epoch, best_val_loss, patience_counter = load_checkpoint_if_exists(best_path, model, optimizer, device)

    for epoch in range(start_epoch, args.epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True, target_util=args.max_gpu_util)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False, target_util=args.max_gpu_util)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, patience_counter=0)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    model.eval()

    ious = []
    with torch.no_grad():
        for image, mask in test_loader:
            image, mask = image.to(device), mask.to(device)
            image_p, (h, w) = pad_to_multiple(image)
            pred = model(image_p)[:, :, :h, :w]
            ious.append(LABEL_SPEC.iou_fn(pred, mask))
    print(f"test mean IoU: {np.mean(ious):.4f}")

    out_path = RESULTS_DIR / "sample_prediction.png"
    save_sample_prediction(model, test_ds, device, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

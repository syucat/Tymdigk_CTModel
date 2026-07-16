"""
Deep Watershed Transform (Bai & Urtasun, CVPR 2017)のこのプロジェクトへの適応版。
NOW.md「次の方向性」の議論の一環(既存のwatershed系はノイズと本物の接触境界を
区別できないという構造的欠陥があるため、DWTの「境界付近を損失で重点的に学習させる
+ 固定閾値1つでカットする」という設計を試す)。

論文との違い(単純化した点):
- 論文は複数の意味クラス(人・車…)ごとにカットレベルを変えるが、ここは「細胞」
  1クラスのみなのでカットレベルは1つの定数(DWT_CUT_LEVEL)。
- エネルギービンの境界は、論文は「学習データ全体でピクセル数のバランスを取る」と
  書かれているのみで正確な式は不明なため、ここでは「境界からmargin px以内=ビン0」
  以外の前景ピクセルについて、インスタンスごとのmax正規化深さ(既存の
  distance_transformモードと同じ値)を、学習データ全体でプールしてから
  等分位点(quantile)でK-2分割する、という具体的な実装で代用している。
- 損失の重み付けもクラスごとのピクセル数の逆数の平方根で代用している(論文の
  正確な重み式ck は式の詳細が不明だったため)。

パイプライン:
  1. Direction Network (DN)を単体で事前学習(入力=画像1ch、出力=方向ベクトル2ch、
     角度誤差損失)
  2. Watershed Transform Network (WTN)を単体で事前学習(入力=画像+GTの方向3ch、
     出力=K段階のエネルギークラス、重み付きクロスエントロピー損失)
  3. DN+WTNを結合してend-to-endでfine-tuning(DNの予測方向を実際にWTNに渡す)

推論: DN→WTNの順に通し、エネルギークラスをカットレベルで閾値処理→連結成分→
元の前景範囲まで拡張→輪郭抽出。

使い方:
  python3 dwt.py --quick --stage all       # 動作確認(少数フレーム・少数エポック)
  python3 dwt.py --stage 1                 # DNだけ学習
  python3 dwt.py --stage 2                 # WTNだけ学習(GTの方向を使う)
  python3 dwt.py --stage 3                 # 結合fine-tuning
"""

import argparse
import time

import cv2
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage.segmentation import expand_labels
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import main

DWT_N_BINS = 16
DWT_BOUNDARY_MARGIN_PX = 2  # ビン0(背景+境界近傍)とみなす、境界からの生ピクセル距離
DWT_CUT_LEVEL = 1  # 推論時、このレベル以下のクラスを「境界」として削る

DWT_MODELS_DIR = main.Path(__file__).parent / "models" / main.DATASET_NAME / "dwt"
DWT_RESULTS_DIR = main.Path(__file__).parent / "results" / main.DATASET_NAME / "dwt"
DWT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DWT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------- 教師データ ----

def _instance_raw_and_norm_distance(instance_slice):
    """raw_dist: 各前景ピクセルの、自インスタンス境界までの生ピクセル距離。
    norm_depth: インスタンスごとに自身の最大距離で割った0-1の相対深さ
    (既存のdistance_transformモードのdepthチャンネルと同じ定義)。"""
    raw_dist = main.derive_distance_target_2d(instance_slice)
    norm_depth = np.zeros_like(raw_dist)
    for inst_id in np.unique(instance_slice):
        if inst_id == 0:
            continue
        m = instance_slice == inst_id
        mx = raw_dist[m].max()
        if mx > 0:
            norm_depth[m] = raw_dist[m] / mx
    return raw_dist, norm_depth


def direction_build_target(instance_slice):
    """(dy, dx, weight)の3チャンネル。dy,dxはGTの生距離の勾配から導出した単位方向
    ベクトル(既存の_derive_unit_flow_from_depthを再利用)。weightは
    1/sqrt(インスタンス面積)で、小さい細胞が大きい細胞に損失を支配されないようにする
    (論文のarea-based weighting)。"""
    raw_dist = main.derive_distance_target_2d(instance_slice)
    fg = instance_slice > 0
    flow_y, flow_x = main._derive_unit_flow_from_depth(raw_dist, fg)
    weight = np.zeros(instance_slice.shape, dtype=np.float32)
    for inst_id in np.unique(instance_slice):
        if inst_id == 0:
            continue
        m = instance_slice == inst_id
        weight[m] = 1.0 / np.sqrt(m.sum())
    return torch.from_numpy(np.stack([flow_y, flow_x, weight], axis=0).astype(np.float32))


def compute_energy_bin_edges(train_frames, mask_cache, n_bins=DWT_N_BINS,
                              margin_px=DWT_BOUNDARY_MARGIN_PX):
    """ビン1..n_bins-1の境界値(n_bins-2個)を、学習データ全体でプールした
    「境界からmargin px超の前景ピクセルの相対深さ」の等分位点として計算する。"""
    depths = []
    for f in train_frames:
        vol = mask_cache[f]
        slices = [vol[z] for z in range(vol.shape[0])] if main.IS_3D else [vol]
        for sl in slices:
            if not (sl > 0).any():
                continue
            raw_dist, norm_depth = _instance_raw_and_norm_distance(sl)
            beyond = (sl > 0) & (raw_dist > margin_px)
            if beyond.any():
                depths.append(norm_depth[beyond])
    all_depths = np.concatenate(depths)
    quantiles = np.linspace(0, 1, n_bins)[1:-1]
    return np.quantile(all_depths, quantiles)


def energy_build_target(instance_slice, bin_edges, margin_px=DWT_BOUNDARY_MARGIN_PX):
    """各画素をK段階のエネルギークラス(long)に割り当てる。
    ビン0=背景 または 境界からmargin px以内。ビン1..K-1=それ以外の前景を
    bin_edgesで分位点分割。"""
    raw_dist, norm_depth = _instance_raw_and_norm_distance(instance_slice)
    fg = instance_slice > 0
    energy = np.zeros(instance_slice.shape, dtype=np.int64)
    beyond = fg & (raw_dist > margin_px)
    if beyond.any():
        energy[beyond] = np.digitize(norm_depth[beyond], bin_edges) + 1
    return torch.from_numpy(energy)


def compute_class_weights(samples, mask_cache, bin_edges, device, n_bins=DWT_N_BINS):
    """クラスごとのピクセル数の逆数の平方根を重みにする(不均衡補正。
    論文の「低エネルギー帯を重視する」ck の代用、詳細はファイル冒頭docstring参照)。"""
    counts = np.zeros(n_bins, dtype=np.int64)
    for frame_idx, z in samples:
        vol = mask_cache[frame_idx]
        sl = vol[z] if z is not None else vol
        energy = energy_build_target(sl, bin_edges).numpy()
        counts += np.bincount(energy.ravel(), minlength=n_bins)
    counts = np.maximum(counts, 1)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32)).to(device)


class DWTDataset(Dataset):
    def __init__(self, samples, mask_cache, norm_lo, norm_hi, bin_edges):
        self.samples = samples
        self.mask_cache = mask_cache
        self.norm_lo = norm_lo
        self.norm_hi = norm_hi
        self.bin_edges = bin_edges

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_idx, z = self.samples[idx]
        if z is None:
            raw = tifffile.imread(main.raw_path(frame_idx)).astype(np.float32)
            instance_slice = self.mask_cache[frame_idx]
        else:
            raw = tifffile.imread(main.raw_path(frame_idx), key=z).astype(np.float32)
            instance_slice = self.mask_cache[frame_idx][z]

        image = np.clip((raw - self.norm_lo) / (self.norm_hi - self.norm_lo), 0, 1)
        image_t = torch.from_numpy(image).unsqueeze(0).float()
        direction_t = direction_build_target(instance_slice)
        energy_t = energy_build_target(instance_slice, self.bin_edges)
        return image_t, direction_t, energy_t


# --------------------------------------------------------------- 損失関数 ----

def direction_loss(pred, target):
    """pred: (B,2,H,W)、tanh出力。target: (B,3,H,W) = (dy,dx,weight)。
    角度誤差の二乗をweightで重み付け平均(論文のl_direction)。"""
    pred_y, pred_x = pred[:, 0], pred[:, 1]
    tgt_y, tgt_x, weight = target[:, 0], target[:, 1], target[:, 2]
    fg_mask = weight > 0
    if not fg_mask.any():
        return torch.zeros((), device=pred.device)
    pred_mag = torch.sqrt(pred_y ** 2 + pred_x ** 2).clamp(min=1e-6)
    cos_sim = (pred_y * tgt_y + pred_x * tgt_x) / pred_mag
    cos_sim = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
    angle_err = torch.acos(cos_sim) ** 2
    return (angle_err[fg_mask] * weight[fg_mask]).sum() / weight[fg_mask].sum()


def energy_loss(logits, target, class_weights):
    return F.cross_entropy(logits, target, weight=class_weights)


# --------------------------------------------------------------- 学習ループ ----

def _throttle(t0, device, target_util):
    """main.pyのrun_epochと同じGPU使用率スロットリング(計算にかかった時間に応じて
    小休止を入れ、平均GPU使用率をtarget_util程度に抑える)。3つのrun_epoch_*で
    共通に使うためここに1箇所だけ定義する。"""
    if target_util is not None and target_util < 1.0:
        if device.type == "cuda":
            torch.cuda.synchronize()
        busy = time.time() - t0
        time.sleep(busy * (1 / target_util - 1))


def run_epoch_dn(model, loader, optimizer, device, train, target_util=None):
    model.train(train)
    total, n = 0.0, 0
    for image, direction_t, _energy_t in loader:
        t0 = time.time()
        image, direction_t = image.to(device), direction_t.to(device)
        image_p, (h, w) = main.pad_to_multiple(image)
        with torch.set_grad_enabled(train):
            pred = model(image_p)[:, :, :h, :w]
            loss = direction_loss(pred, direction_t)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        _throttle(t0, device, target_util)
        total += loss.item() * image.size(0)
        n += image.size(0)
    return total / n


def run_epoch_wtn(model, loader, optimizer, device, train, class_weights, target_util=None):
    model.train(train)
    total, n = 0.0, 0
    for image, direction_t, energy_t in loader:
        t0 = time.time()
        image, direction_t, energy_t = image.to(device), direction_t.to(device), energy_t.to(device)
        gt_direction = direction_t[:, :2]  # weightチャンネルは入力に使わない
        wtn_input = torch.cat([image, gt_direction], dim=1)
        wtn_input_p, (h, w) = main.pad_to_multiple(wtn_input)
        with torch.set_grad_enabled(train):
            logits = model(wtn_input_p)[:, :, :h, :w]
            loss = energy_loss(logits, energy_t, class_weights)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        _throttle(t0, device, target_util)
        total += loss.item() * image.size(0)
        n += image.size(0)
    return total / n


def run_epoch_joint(dn_model, wtn_model, loader, optimizer, device, train, class_weights, target_util=None):
    """DNの予測方向を実際にWTNへ渡し、両方を同時にend-to-endで更新する
    (GTの方向で別々に学習した段階1・2と違い、ここで初めてDNとWTNの整合性が
    直接強制される)。"""
    dn_model.train(train)
    wtn_model.train(train)
    total, n = 0.0, 0
    for image, _direction_t, energy_t in loader:
        t0 = time.time()
        image, energy_t = image.to(device), energy_t.to(device)
        image_p, (h, w) = main.pad_to_multiple(image)
        with torch.set_grad_enabled(train):
            direction_pred = dn_model(image_p)[:, :, :h, :w]
            wtn_input = torch.cat([image, direction_pred], dim=1)
            wtn_input_p, (h2, w2) = main.pad_to_multiple(wtn_input)
            logits = wtn_model(wtn_input_p)[:, :, :h2, :w2]
            loss = energy_loss(logits, energy_t, class_weights)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        _throttle(t0, device, target_util)
        total += loss.item() * image.size(0)
        n += image.size(0)
    return total / n


def _train_loop(model, optimizer, train_loader, val_loader, device, args, ckpt_path, run_epoch_fn):
    start_epoch, best_val_loss, patience_counter = main.load_checkpoint_if_exists(
        ckpt_path, model, optimizer, device)
    for epoch in range(start_epoch, args.epochs):
        train_loss = run_epoch_fn(model, train_loader, optimizer, device, True)
        val_loss = run_epoch_fn(model, val_loader, optimizer, device, False)
        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            main.save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_loss, patience_counter=0)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping")
                break
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])


# ------------------------------------------------------------------ 推論 ----

def to_contour_input(dn_model, wtn_model, image_t, device, cut_level=DWT_CUT_LEVEL):
    """image_t: (1,1,H,W)の正規化済み入力(パディング前)。
    DN→WTNの順に通し、エネルギークラスをcut_level以下で切り取って連結成分を取り、
    元の前景範囲(エネルギークラス>0)まで領域を拡張してから輪郭を抽出する
    (論文の「カット→dilateして復元→連結成分」に相当。ここではdilateの代わりに
    skimage.segmentation.expand_labelsで前景内に限定して拡張する)。
    (輪郭リスト, 表示用エネルギークラスmap)を返す。"""
    image_t = image_t.to(device)
    image_p, (h, w) = main.pad_to_multiple(image_t)
    with torch.no_grad():
        direction_pred = dn_model(image_p)[:, :, :h, :w]
        wtn_input = torch.cat([image_t, direction_pred], dim=1)
        wtn_input_p, (h2, w2) = main.pad_to_multiple(wtn_input)
        logits = wtn_model(wtn_input_p)[:, :, :h2, :w2]
    energy_class = logits.argmax(dim=1)[0].cpu().numpy()

    fg_mask = energy_class > 0
    cut_mask = energy_class > cut_level
    labels, n_labels = ndimage.label(cut_mask)
    if n_labels == 0:
        return [], energy_class

    expanded = expand_labels(labels, distance=max(energy_class.shape))
    expanded[~fg_mask] = 0

    contours = []
    for label_id in range(1, int(expanded.max()) + 1):
        binary = (expanded == label_id).astype(np.uint8)
        contours.extend(main.get_contours(binary))
    return contours, energy_class


def save_sample_prediction(dn_model, wtn_model, dataset, device, path, cut_level=DWT_CUT_LEVEL):
    image_t, _direction_t, energy_gt = dataset[0]
    contours, energy_class = to_contour_input(dn_model, wtn_model, image_t.unsqueeze(0), device, cut_level)

    contour_img = cv2.cvtColor((image_t.numpy()[0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_t.numpy()[0], cmap="gray")
    axes[0].set_title("input")
    axes[1].imshow(energy_gt.numpy(), cmap="viridis", vmin=0, vmax=DWT_N_BINS - 1)
    axes[1].set_title("GT energy bins")
    axes[2].imshow(energy_class, cmap="viridis", vmin=0, vmax=DWT_N_BINS - 1)
    axes[2].set_title("predicted energy bins")
    axes[3].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"contours (cut_level={cut_level}, n={len(contours)})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# -------------------------------------------------------------------- CLI ----

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--joint-batch-size", type=int, default=None,
                         help="Stage3(結合fine-tuning)専用のバッチサイズ。DN+WTN両方のactivationを"
                              "同時に保持するため単体学習よりメモリを食う。未指定なら--batch-sizeの半分")
    parser.add_argument("--quick", action="store_true", help="少数フレーム・少数エポックで動作確認する")
    parser.add_argument("--cut-level", type=int, default=DWT_CUT_LEVEL)
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all",
                         help="1=DNのみ, 2=WTNのみ(GTの方向を使用), 3=結合fine-tuning, all=1→2→3を順に実行")
    parser.add_argument("--max-vram-gb", type=float, default=None,
                         help="このプロセスが使うVRAMの上限(GB)。DN+WTNの2ネットワーク分のメモリを"
                              "同時に確保するため、既存の単一ネットワークのモードより消費量が大きい"
                              "(main.pyと同じ仕組み)")
    parser.add_argument("--max-gpu-util", type=float, default=None,
                         help="平均GPU使用率の目標値(0-1)。main.pyと同じ仕組み")
    args = parser.parse_args()
    if args.quick and args.epochs == 100:
        args.epochs = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if device.type == "cuda" and args.max_vram_gb is not None:
        total_mem = torch.cuda.get_device_properties(0).total_memory
        fraction = min((args.max_vram_gb * 1024 ** 3) / total_mem, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        print(f"VRAM上限: {args.max_vram_gb}GB (device全体の{fraction * 100:.1f}%)に設定")

    train_frames, val_frames, test_frames = main.build_frame_split()
    if args.quick:
        train_frames, val_frames, test_frames = train_frames[:3], val_frames[:2], test_frames[:1]
    print(f"frames: train={len(train_frames)} val={len(val_frames)} test={len(test_frames)}")

    norm_lo, norm_hi = main.compute_norm_bounds(train_frames)

    train_samples, train_mask_cache = main.build_samples_with_mask_cache(train_frames)
    val_samples, val_mask_cache = main.build_samples_with_mask_cache(val_frames)
    print(f"samples: train={len(train_samples)} val={len(val_samples)}")

    print("エネルギービンの境界・クラス重みを計算中...")
    bin_edges = compute_energy_bin_edges(train_frames, train_mask_cache)
    class_weights = compute_class_weights(train_samples, train_mask_cache, bin_edges, device)
    print(f"bin_edges={np.round(bin_edges, 3)}")
    print(f"class_weights={np.round(class_weights.cpu().numpy(), 3)}")

    train_ds = DWTDataset(train_samples, train_mask_cache, norm_lo, norm_hi, bin_edges)
    val_ds = DWTDataset(val_samples, val_mask_cache, norm_lo, norm_hi, bin_edges)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    joint_batch_size = args.joint_batch_size or max(1, args.batch_size // 2)
    joint_train_loader = DataLoader(train_ds, batch_size=joint_batch_size, shuffle=True, num_workers=0)
    joint_val_loader = DataLoader(val_ds, batch_size=joint_batch_size, shuffle=False, num_workers=0)

    dn_model = main.UNet(in_channels=1, out_channels=2, activation="tanh").to(device)
    wtn_model = main.UNet(in_channels=3, out_channels=DWT_N_BINS, activation="none").to(device)

    dn_path = DWT_MODELS_DIR / "dn_best.pth"
    wtn_path = DWT_MODELS_DIR / "wtn_best.pth"
    joint_path = DWT_MODELS_DIR / "joint_best.pth"

    if args.stage in ("1", "all"):
        print("=== Stage 1: Direction Network (単体、角度誤差損失) ===")
        opt1 = torch.optim.Adam(dn_model.parameters(), lr=1e-4)
        _train_loop(dn_model, opt1, train_loader, val_loader, device, args, dn_path,
                    lambda m, l, o, dv, tr: run_epoch_dn(m, l, o, dv, tr, args.max_gpu_util))
    elif dn_path.exists():
        dn_model.load_state_dict(torch.load(dn_path, map_location=device)["model"])

    if args.stage in ("2", "all"):
        print("=== Stage 2: Watershed Transform Network (単体、GTの方向を使用) ===")
        opt2 = torch.optim.Adam(wtn_model.parameters(), lr=1e-4)
        _train_loop(wtn_model, opt2, train_loader, val_loader, device, args, wtn_path,
                    lambda m, l, o, dv, tr: run_epoch_wtn(m, l, o, dv, tr, class_weights, args.max_gpu_util))
    elif wtn_path.exists():
        wtn_model.load_state_dict(torch.load(wtn_path, map_location=device)["model"])

    if args.stage in ("3", "all"):
        print("=== Stage 3: DN+WTN 結合end-to-end fine-tuning ===")
        joint_model = nn.ModuleDict({"dn": dn_model, "wtn": wtn_model})
        opt3 = torch.optim.Adam(joint_model.parameters(), lr=1e-5)
        _train_loop(joint_model, opt3, joint_train_loader, joint_val_loader, device, args, joint_path,
                    lambda m, l, o, dv, tr: run_epoch_joint(m["dn"], m["wtn"], l, o, dv, tr, class_weights, args.max_gpu_util))
    elif joint_path.exists():
        ckpt = torch.load(joint_path, map_location=device)
        joint_model = nn.ModuleDict({"dn": dn_model, "wtn": wtn_model})
        joint_model.load_state_dict(ckpt["model"])

    dn_model.eval()
    wtn_model.eval()

    test_samples, test_mask_cache = main.build_samples_with_mask_cache(test_frames)
    test_ds = DWTDataset(test_samples, test_mask_cache, norm_lo, norm_hi, bin_edges)
    if len(test_ds) > 0:
        out_path = DWT_RESULTS_DIR / "sample_prediction.png"
        save_sample_prediction(dn_model, wtn_model, test_ds, device, out_path, args.cut_level)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main_cli()

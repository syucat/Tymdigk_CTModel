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
import random
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "DATA_CELL" / "Fluo-N3DH-SIM+"
SEQ = "01"
N_FRAMES = 150
RESULTS_DIR = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

SEED = 0
PAD_TO = 8  # 3回のMaxPool(2x2)で割り切れるように入力をパディングする単位

PERCENTILE_LO, PERCENTILE_HI = 0.5, 99.5
N_FRAMES_FOR_NORM_STATS = 30  # 正規化の統計を計算するのに使うtrainフレーム数


def raw_path(frame_idx):
    return DATA_DIR / SEQ / f"t{frame_idx:03d}.tif"


def mask_path(frame_idx):
    return DATA_DIR / f"{SEQ}_GT" / "SEG" / f"man_seg{frame_idx:03d}.tif"


def build_samples_with_mask_cache(frame_list):
    """
    各フレームのGTマスク全体を1回だけ読み込み、細胞が写っているzスライスの
    (frame_idx, z)サンプルリストと、二値化マスクのキャッシュ(dict)を同時に作る。
    学習中に同じマスクをディスクから読み直さずに済むようにするため。
    """
    samples = []
    mask_cache = {}
    for f in frame_list:
        mask = tifffile.imread(mask_path(f))
        binary_mask = (mask > 0)
        nonzero = np.where(binary_mask.sum(axis=(1, 2)) > 0)[0]
        if len(nonzero) == 0:
            continue
        mask_cache[f] = binary_mask
        for z in range(int(nonzero.min()), int(nonzero.max()) + 1):
            samples.append((f, z))
    return samples, mask_cache


def build_frame_split(seed=SEED):
    frames = list(range(N_FRAMES))
    rng = random.Random(seed)
    rng.shuffle(frames)
    n_train = int(N_FRAMES * 0.7)
    n_val = int(N_FRAMES * 0.2)
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
        # raw画像はkey=zでそのページだけ読む（ボリューム全体を読み込まない）
        # maskは事前にbuild_samples_with_mask_cacheでキャッシュ済みのものを使う（再読み込みしない）
        raw = tifffile.imread(raw_path(frame_idx), key=z).astype(np.float32)
        binary_mask = self.mask_cache[frame_idx][z].astype(np.float32)

        image = np.clip((raw - self.norm_lo) / (self.norm_hi - self.norm_lo), 0, 1)
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(binary_mask).unsqueeze(0)


# ---------------------------------------------------------------- U-Net ----

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = conv_block(1, 64)
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

        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


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


def iou_score(pred_binary, target, eps=1e-6):
    pred_binary = pred_binary.reshape(pred_binary.size(0), -1)
    target = target.reshape(target.size(0), -1)
    intersection = (pred_binary * target).sum(dim=1)
    union = ((pred_binary + target) > 0).float().sum(dim=1)
    return ((intersection + eps) / (union + eps)).mean().item()


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
            loss = bce_dice_loss(pred, mask)
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
    pred_binary = (pred > 0.5).float().cpu().numpy()[0, 0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image.numpy()[0], cmap="gray")
    axes[0].set_title("input")
    axes[1].imshow(mask.numpy()[0], cmap="gray")
    axes[1].set_title("GT mask")
    axes[2].imshow(pred_binary, cmap="gray")
    axes[2].set_title("predicted mask")
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

    model = UNet().to(device)
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
            pred_binary = (pred > 0.5).float()
            ious.append(iou_score(pred_binary, mask))
    print(f"test mean IoU: {np.mean(ious):.4f}")

    out_path = RESULTS_DIR / "sample_prediction.png"
    save_sample_prediction(model, test_ds, device, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

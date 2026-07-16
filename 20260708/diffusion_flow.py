"""
熱拡散シミュレーション(Cellpose式、設計B)による(前景, dy, dx)学習パイプライン。
NEXTOBJECTIVE.md「今すぐやること」参照。

設計の要点(ユーザーとの議論で確定した内容):
- 学習データはFluo-N3DH-CHOのST(Silver Truth)を使う。学習はSTで行うが、
  評価は本物のGT(19スライス、eval_real_gt.py)で行う。
- flow場の教師データは、GT/STのインスタンスマスクの中で熱拡散シミュレーションを
  行い、収束しきる前(反復回数=2*dt.max()^2)で打ち切った時点の温度分布の勾配。
  収束しきると全体が同じ温度になり意味のある勾配が消えるため、あえて途中で止める。
- 距離変換の勾配(設計A、既存のdistance_flow_trained)と違い、熱拡散は前景マスクの
  形だけから事後的に計算し直すことができない(接触した2細胞は前景マスク上では
  1つの塊にしか見えず、インスタンスごとに区切れないため)。モデルは生画像の
  輝度・テクスチャの手がかりを使って直接(前景,dy,dx)を予測する必要がある。
- 熱拡散シミュレーション自体は教師データを作るための計算であり、モデルが学習する
  わけではない。モデルの入力は生画像のみ、熱拡散は一度も呼ばれない。

計算コストの都合上、flow場はデータセット全体に対して事前に1回だけ計算しキャッシュし、
学習時はキャッシュを読み込むだけにする(`precompute_cache()`)。
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader

import main

CACHE_DIR = main.Path(__file__).parent / "cache" / main.DATASET_NAME / "diffusion_flow"
DIFFUSION_FLOW_MODELS_DIR = main.Path(__file__).parent / "models" / main.DATASET_NAME / "diffusion_flow"
DIFFUSION_ITER_COEF = 2.0  # n_iter = DIFFUSION_ITER_COEF * dt.max()^2
DIFFUSION_MAX_ITER = 6000  # 安全弁(通常はdt.max()由来の値がこれより小さい)


# ------------------------------------------------------------- 熱拡散計算 ----

def compute_diffusion_flow_2d(instance_slice, iter_coef=DIFFUSION_ITER_COEF,
                               max_iter=DIFFUSION_MAX_ITER):
    """2DのインスタンスIDラベル画像から、熱拡散ベースの(flow_y, flow_x)を計算する。

    全インスタンスを1回のループで同時に処理する: 各ピクセルは自分と同じ
    インスタンスIDの隣接ピクセルとだけ値を平均する(拡散が別インスタンストに
    漏れない)。各インスタンスは自分のdt.max()^2に比例した反復回数に達したら
    それ以降は更新を止める(収束しきって勾配が消えるのを防ぐ、詳細はファイル
    冒頭docstring参照)。
    """
    h, w = instance_slice.shape
    fg = instance_slice > 0
    if not fg.any():
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    T = np.zeros((h, w), dtype=np.float32)
    target_iter = np.zeros((h, w), dtype=np.int64)
    source_mask = np.zeros((h, w), dtype=bool)

    for inst_id in np.unique(instance_slice):
        if inst_id == 0:
            continue
        mask = instance_slice == inst_id
        dt = ndimage.distance_transform_edt(mask)
        n_iter = min(int(iter_coef * dt.max() ** 2), max_iter)
        target_iter[mask] = max(n_iter, 1)
        cy, cx = np.unravel_index(np.argmax(dt), dt.shape)
        source_mask[cy, cx] = True

    # ラベルのシフト版は反復中に変わらないので、ループの外で1回だけ計算しておく
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    shifted_labels = [np.roll(instance_slice, (dy, dx), axis=(0, 1)) for dy, dx in shifts]
    same_label_masks = [(sl == instance_slice) & fg for sl in shifted_labels]

    max_n_iter = int(target_iter.max())
    for it in range(max_n_iter):
        still_active = (target_iter > it) & fg
        if not still_active.any():
            break
        neighbor_sum = np.zeros((h, w), dtype=np.float32)
        neighbor_count = np.zeros((h, w), dtype=np.float32)
        for (dy, dx), same_label in zip(shifts, same_label_masks):
            shifted_T = np.roll(T, (dy, dx), axis=(0, 1))
            neighbor_sum += np.where(same_label, shifted_T, 0.0)
            neighbor_count += same_label
        new_T = neighbor_sum / np.maximum(neighbor_count, 1.0)
        new_T[source_mask] = 1.0
        T = np.where(still_active, new_T, T)

    grad_y, grad_x = np.gradient(T)
    mag = np.sqrt(grad_y ** 2 + grad_x ** 2)
    flow_y = np.zeros((h, w), dtype=np.float32)
    flow_x = np.zeros((h, w), dtype=np.float32)
    valid = fg & (mag > 1e-8)
    flow_y[valid] = grad_y[valid] / mag[valid]
    flow_x[valid] = grad_x[valid] / mag[valid]
    return flow_y, flow_x


# --------------------------------------------------------------- キャッシュ ----

def cache_path(frame_idx, z):
    z_part = f"_z{z:03d}" if z is not None else ""
    return CACHE_DIR / f"t{frame_idx:03d}{z_part}.npy"


def precompute_cache(frame_list, mask_cache, force=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    samples, _ = main.build_samples_with_mask_cache(frame_list)
    t0 = time.time()
    for i, (frame_idx, z) in enumerate(samples):
        path = cache_path(frame_idx, z)
        if path.exists() and not force:
            continue
        vol = mask_cache[frame_idx]
        instance_slice = vol[z] if z is not None else vol
        flow_y, flow_x = compute_diffusion_flow_2d(instance_slice)
        np.save(path, np.stack([flow_y, flow_x], axis=0).astype(np.float32))
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(samples)}  経過{elapsed:.0f}秒  "
                  f"(平均{elapsed / (i + 1):.2f}秒/枚)")
    print(f"キャッシュ完了: {len(samples)}枚  {CACHE_DIR}")


def load_cached_flow(frame_idx, z):
    arr = np.load(cache_path(frame_idx, z))
    return arr[0], arr[1]


# --------------------------------------------------------------- Dataset ----

class DiffusionFlowDataset(Dataset):
    def __init__(self, samples, mask_cache, norm_lo, norm_hi):
        self.samples = samples
        self.mask_cache = mask_cache
        self.norm_lo = norm_lo
        self.norm_hi = norm_hi

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

        fg = (instance_slice > 0).astype(np.float32)
        flow_y, flow_x = load_cached_flow(frame_idx, z)
        target_t = torch.from_numpy(np.stack([fg, flow_y, flow_x], axis=0).astype(np.float32))
        return image_t, target_t


# --------------------------------------------------------------- 損失関数 ----

def diffusion_flow_loss(pred, target):
    """pred, target: (B,3,H,W) = (前景, dy, dx)。前景はbce_dice_loss、
    dy,dxは前景マスクしたMSE(既存のdistance_flow_trained_lossと同じパターン、
    深さチャンネルが無い分シンプル)。"""
    fg_pred, dy_pred, dx_pred = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    fg_target, dy_target, dx_target = target[:, 0:1], target[:, 1:2], target[:, 2:3]
    fg_loss = main.bce_dice_loss(fg_pred, fg_target)
    fg_mask = fg_target > 0
    if fg_mask.any():
        dy_loss = F.mse_loss(dy_pred[fg_mask], dy_target[fg_mask])
        dx_loss = F.mse_loss(dx_pred[fg_mask], dx_target[fg_mask])
    else:
        dy_loss = dx_loss = torch.zeros((), device=pred.device)
    return fg_loss + dy_loss + dx_loss


def diffusion_flow_iou(pred, target):
    return main.iou_score((pred[:, 0:1] > 0.5).float(), target[:, 0:1])


# ------------------------------------------------------------------ 推論 ----

def to_contour_input(pred, image=None, close_radius=0):
    """pred: (1,3,H,W) = (前景sigmoid, dy tanh, dx tanh)。既存のdistance_flow_trained
    と同じ軌跡積分+クラスタリング(main._integrate_and_cluster)を再利用する。
    深さチャンネルが無いので表示用マップは前景確率をそのまま使う。

    close_radius>0の場合、前景マスクにモルフォロジー・クロージングをかけてから
    軌跡積分する。細胞内部の暗い模様が前景確率マップに小さな"穴"を作り、
    その穴のせいで軌跡積分が1つの細胞を2つのクラスタに分けてしまう問題への対策
    (2026-07-16、t051で実際に確認された失敗例。再学習不要、後処理のみで対応)。"""
    fg_prob = pred[0, 0].cpu().numpy()
    flow_y = pred[0, 1].cpu().numpy()
    flow_x = pred[0, 2].cpu().numpy()
    fg_mask = fg_prob > 0.5

    if close_radius > 0:
        k = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        fg_mask = cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    labels = main._integrate_and_cluster(fg_mask, flow_y, flow_x)
    contours = []
    for label_id in range(1, labels.max() + 1):
        binary = (labels == label_id).astype(np.uint8)
        contours.extend(main.get_contours(binary))
    return contours, fg_prob


# -------------------------------------------------------------------- CLI ----

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute-only", action="store_true",
                         help="キャッシュの事前計算だけ行って終了する(学習しない)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-vram-gb", type=float, default=None)
    parser.add_argument("--max-gpu-util", type=float, default=None)
    args = parser.parse_args()
    if args.quick and args.epochs == 100:
        args.epochs = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_frames, val_frames, test_frames = main.build_frame_split()
    if args.quick:
        train_frames, val_frames, test_frames = train_frames[:3], val_frames[:2], test_frames[:1]
    print(f"frames: train={len(train_frames)} val={len(val_frames)} test={len(test_frames)}")

    norm_lo, norm_hi = main.compute_norm_bounds(train_frames)
    train_samples, train_mask_cache = main.build_samples_with_mask_cache(train_frames)
    val_samples, val_mask_cache = main.build_samples_with_mask_cache(val_frames)
    print(f"samples: train={len(train_samples)} val={len(val_samples)}")

    print("熱拡散flow場のキャッシュを計算中(初回のみ、以降はキャッシュを再利用)...")
    precompute_cache(train_frames, train_mask_cache)
    precompute_cache(val_frames, val_mask_cache)
    if args.precompute_only:
        return

    train_ds = DiffusionFlowDataset(train_samples, train_mask_cache, norm_lo, norm_hi)
    val_ds = DiffusionFlowDataset(val_samples, val_mask_cache, norm_lo, norm_hi)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if device.type == "cuda" and args.max_vram_gb is not None:
        total_mem = torch.cuda.get_device_properties(0).total_memory
        fraction = min((args.max_vram_gb * 1024 ** 3) / total_mem, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        print(f"VRAM上限: {args.max_vram_gb}GB (device全体の{fraction * 100:.1f}%)に設定")

    model = main.UNet(out_channels=3, activation="sigmoid_tanh2").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    DIFFUSION_FLOW_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = DIFFUSION_FLOW_MODELS_DIR / "unet_best.pth"
    start_epoch, best_val_loss, patience_counter = main.load_checkpoint_if_exists(
        best_path, model, optimizer, device)

    for epoch in range(start_epoch, args.epochs):
        model.train(True)
        total, n = 0.0, 0
        for image, target in train_loader:
            t0 = time.time()
            image, target = image.to(device), target.to(device)
            image_p, (h, w) = main.pad_to_multiple(image)
            pred = model(image_p)[:, :, :h, :w]
            loss = diffusion_flow_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.max_gpu_util is not None and args.max_gpu_util < 1.0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                busy = time.time() - t0
                time.sleep(busy * (1 / args.max_gpu_util - 1))
            total += loss.item() * image.size(0)
            n += image.size(0)
        train_loss = total / n

        model.train(False)
        total, n = 0.0, 0
        with torch.no_grad():
            for image, target in val_loader:
                image, target = image.to(device), target.to(device)
                image_p, (h, w) = main.pad_to_multiple(image)
                pred = model(image_p)[:, :, :h, :w]
                loss = diffusion_flow_loss(pred, target)
                total += loss.item() * image.size(0)
                n += image.size(0)
        val_loss = total / n

        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            main.save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, patience_counter=0)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping")
                break

    print(f"完了: {best_path}")


if __name__ == "__main__":
    main_cli()

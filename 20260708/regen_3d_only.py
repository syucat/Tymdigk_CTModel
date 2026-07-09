"""
view3d.html だけを再生成する一時スクリプト(カメラ同期JSを反映するため)。
2Dパネルの画像(z*.png)はそのまま再利用し、再計算しない。
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch

from main import build_frame_split, compute_norm_bounds, mask_path, raw_path
from track_and_visualize import load_model, predict_slice, track_contours_across_z, render_3d_view, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.model_path), device)

    train_frames, val_frames, test_frames = build_frame_split()
    norm_lo, norm_hi = compute_norm_bounds(train_frames)

    out_dir = RESULTS_DIR / args.run_name

    for f in test_frames:
        mask_volume = tifffile.imread(mask_path(f))
        nonzero = np.where(mask_volume.sum(axis=(1, 2)) > 0)[0]
        if len(nonzero) == 0:
            continue
        z_range = list(range(int(nonzero.min()), int(nonzero.max()) + 1))

        binary_masks = []
        for z in z_range:
            raw = tifffile.imread(raw_path(f), key=z)
            binary, _ = predict_slice(model, raw, norm_lo, norm_hi, device)
            binary_masks.append(binary)

        per_slice_tagged, _, _ = track_contours_across_z(binary_masks)

        frame_dir = out_dir / f"t{f:03d}"
        render_3d_view(z_range, mask_volume, per_slice_tagged, frame_dir / "view3d.html", "../../plotly_bundle.js")
        print(f"t{f:03d} view3d.html updated")


if __name__ == "__main__":
    main()

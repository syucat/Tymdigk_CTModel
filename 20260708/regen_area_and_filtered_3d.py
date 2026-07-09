"""
既存のrun(results/<run_name>/)に対して、面積ヒストグラム(フィルタ前後)と
面積閾値フィルタ後の3Dビューだけを追加生成し、manifest.jsonにフィールドを
追記する一時スクリプト。2Dの7パネルPNG(z*.png)とview3d.htmlは再生成しない
(再計算に時間がかかる7パネル描画を避けるため)。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import torch

from main import build_frame_split, compute_norm_bounds, mask_path, raw_path
from track_and_visualize import (
    RESULTS_DIR, load_model, predict_slice, track_contours_across_z,
    compute_tag_avg_areas, compute_gt_avg_areas, filter_tags_by_area,
    render_area_histogram, render_3d_view, NOISE_AREA_THRESHOLD,
)


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
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames_by_id = {fm["frame"]: fm for fm in manifest["frames"]}

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

        tag_areas = compute_tag_avg_areas(per_slice_tagged)
        gt_areas = compute_gt_avg_areas(mask_volume, z_range)
        filtered_tagged, keep_ids = filter_tags_by_area(per_slice_tagged, tag_areas, NOISE_AREA_THRESHOLD)
        filtered_tag_areas = {t: a for t, a in tag_areas.items() if t in keep_ids}

        render_area_histogram(tag_areas, filtered_tag_areas, gt_areas, NOISE_AREA_THRESHOLD,
                               frame_dir / "area_hist.png")
        render_3d_view(z_range, mask_volume, filtered_tagged, frame_dir / "view3d_filtered.html",
                        "../../plotly_bundle.js",
                        pred_title=f"predicted tags (3D, area filter={NOISE_AREA_THRESHOLD})")

        fm = frames_by_id[f]
        fm["nPredFiltered"] = len(keep_ids)
        fm["noiseAreaThreshold"] = NOISE_AREA_THRESHOLD
        fm["view3dFiltered"] = f"t{f:03d}/view3d_filtered.html"
        fm["areaHist"] = f"t{f:03d}/area_hist.png"
        fm.pop("areaHistBefore", None)
        fm.pop("areaHistAfter", None)

        print(f"t{f:03d}: pred={fm['nPredTotal']} -> filtered={fm['nPredFiltered']} (GT={fm['nGtTotal']})")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(f"updated: {manifest_path}")


if __name__ == "__main__":
    main()

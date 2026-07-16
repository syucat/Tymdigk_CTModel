"""
track_and_visualize.pyのCellpose版。事前学習済みのCellpose(4.x, cpsam_v2、
Cellpose-SAM)をゼロショット(このデータセットでの再学習なし)でそのまま使い、
輪郭を取得する。DWT等の自前手法と同じ土俵で比較できるよう、Zスライス間タグ付け・
面積ノイズフィルタ・描画はtrack_and_visualize.pyの既存関数をそのまま再利用し、
「輪郭をどう作るか」の部分だけをCellposeの`model.eval()`に差し替える
(dwt_track_and_visualize.pyと同じ構成)。

Cellpose自体の学習方法(内部でどう学習されたか)はブラックボックスとして扱い、
ここでは「学習済みの汎用モデルとしてどれだけこのデータに効くか」だけを見る。
"""

import argparse

import cv2
import numpy as np
import tifffile

from main import (
    RESULTS_DIR,
    build_frame_split, compute_norm_bounds, compute_abs_bounds,
    raw_path, mask_path, get_contours,
)
from track_and_visualize import (
    track_contours_across_z, compute_tag_avg_areas, compute_gt_avg_areas,
    filter_tags_by_area, render_area_histogram, render_slice_image, render_3d_view,
    NOISE_AREA_THRESHOLD,
)

OUTPUT_TITLE = "output (Cellpose mask, binarized)"
OUTPUT_CMAP = "gray"
OUTPUT_VMAX = 1.0


def predict_slice_cellpose(model, raw_slice, diameter):
    """(輪郭リスト, 表示用0/1マップ)を返す。Cellposeは正規化・チャンネル判定を
    内部で行うので、生の輝度値をそのまま渡す(このプロジェクトの他モデルのような
    パーセンタイル正規化は不要、というよりCellpose側の想定と混ぜない方が安全)。"""
    masks, _flows, _styles = model.eval(raw_slice, diameter=diameter)
    contours = []
    for label_id in range(1, int(masks.max()) + 1):
        binary = (masks == label_id).astype(np.uint8)
        contours.extend(get_contours(binary))
    return contours, (masks > 0).astype(np.float32)


def process_frame(model, frame_idx, abs_lo, abs_hi, norm_lo, norm_hi, diameter, out_dir, plotly_bundle_relpath):
    mask_volume = tifffile.imread(mask_path(frame_idx))
    nonzero = np.where(mask_volume.sum(axis=(1, 2)) > 0)[0]
    if len(nonzero) == 0:
        return None
    z_range = list(range(int(nonzero.min()), int(nonzero.max()) + 1))
    true_cell_ids = set(np.unique(mask_volume[z_range])) - {0}

    raw_slices, contours_per_slice, out_maps = [], [], []
    for z in z_range:
        raw = tifffile.imread(raw_path(frame_idx), key=z)
        raw_slices.append(raw)
        contours, out_map = predict_slice_cellpose(model, raw, diameter)
        contours_per_slice.append(contours)
        out_maps.append(out_map)

    per_slice_tagged, cumulative_tags, total_tags = track_contours_across_z(
        contours_per_slice, raw_slices[0].shape)

    frame_dir = out_dir / f"t{frame_idx:03d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    slices_meta = []
    for i, z in enumerate(z_range):
        fname = f"z{z:03d}.png"
        render_slice_image(raw_slices[i], mask_volume[z], out_maps[i], per_slice_tagged[i],
                            abs_lo, abs_hi, norm_lo, norm_hi, frame_dir / fname,
                            output_title=OUTPUT_TITLE, output_cmap=OUTPUT_CMAP, output_vmax=OUTPUT_VMAX)
        n_gt_slice = len(set(np.unique(mask_volume[z])) - {0})
        slices_meta.append({
            "z": z, "file": f"t{frame_idx:03d}/{fname}",
            "nGt": n_gt_slice, "nPred": len(per_slice_tagged[i]), "cum": cumulative_tags[i],
        })

    render_3d_view(z_range, mask_volume, per_slice_tagged, frame_dir / "view3d.html", plotly_bundle_relpath)

    tag_areas = compute_tag_avg_areas(per_slice_tagged)
    gt_areas = compute_gt_avg_areas(mask_volume, z_range)
    filtered_tagged, keep_ids = filter_tags_by_area(per_slice_tagged, tag_areas, NOISE_AREA_THRESHOLD)
    filtered_tag_areas = {t: a for t, a in tag_areas.items() if t in keep_ids}

    render_area_histogram(tag_areas, filtered_tag_areas, gt_areas, NOISE_AREA_THRESHOLD,
                           frame_dir / "area_hist.png")
    render_3d_view(z_range, mask_volume, filtered_tagged, frame_dir / "view3d_filtered.html",
                    plotly_bundle_relpath,
                    pred_title=f"predicted tags (3D, area filter={NOISE_AREA_THRESHOLD})")

    return {
        "frame": frame_idx, "nGtTotal": len(true_cell_ids), "nPredTotal": total_tags,
        "nPredFiltered": len(keep_ids), "noiseAreaThreshold": NOISE_AREA_THRESHOLD,
        "slices": slices_meta, "view3d": f"t{frame_idx:03d}/view3d.html",
        "view3dFiltered": f"t{frame_idx:03d}/view3d_filtered.html",
        "areaHist": f"t{frame_idx:03d}/area_hist.png",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diameter", type=float, default=None,
                         help="細胞の直径(px)のヒント。Noneなら自動推定(Cellpose-SAMはスケールに"
                              "比較的頑健なので通常はNoneのままでよい)")
    parser.add_argument("--run-name", default="1epoch_cellpose",
                         help="results/以下に作る実験フォルダ名(再学習していないので'1epoch'は仮の"
                              "ラベルだが、①モデル列の学習段階グルーピングに合わせるため接頭辞を揃える)")
    parser.add_argument("--description", default=(
        "Cellpose(Stringer et al.、cpsam_v2/Cellpose-SAM)の事前学習済みモデルをゼロショットで"
        "適用(このデータセットでの再学習なし)。輪郭の作り方だけをCellposeに差し替え、"
        "Zスライス間タグ付け・面積ノイズフィルタは他手法と共通。"))
    parser.add_argument("--model-label", default="Cellpose (事前学習済み, ゼロショット)")
    parser.add_argument("--contour-label", default="Cellpose標準出力")
    args = parser.parse_args()

    from cellpose import models
    model = models.CellposeModel(gpu=True)

    train_frames, val_frames, test_frames = build_frame_split()
    norm_lo, norm_hi = compute_norm_bounds(train_frames)
    abs_lo, abs_hi = compute_abs_bounds(train_frames)

    import plotly.offline as pyo
    (RESULTS_DIR / "plotly_bundle.js").write_text(pyo.get_plotlyjs(), encoding="utf-8")

    out_dir = RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_meta = []
    for f in test_frames:
        print(f"processing t{f:03d} ...")
        result = process_frame(model, f, abs_lo, abs_hi, norm_lo, norm_hi, args.diameter,
                                out_dir, "../../plotly_bundle.js")
        if result is not None:
            frames_meta.append(result)
            print(f"  GT={result['nGtTotal']} pred={result['nPredTotal']} "
                  f"pred_filtered={result['nPredFiltered']} slices={len(result['slices'])}")

    manifest = {
        "run_name": args.run_name, "description": args.description,
        "model_label": args.model_label, "contour_label": args.contour_label,
        "frames": frames_meta,
    }
    import json
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (out_dir / "description.txt").write_text(args.description, encoding="utf-8")
    print(f"saved: {out_dir}/manifest.json ({len(frames_meta)} frames)")


if __name__ == "__main__":
    main()

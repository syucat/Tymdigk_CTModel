"""
track_and_visualize.pyのbinary_DiffusionFlow(diffusion_flow.py)版。
dwt_track_and_visualize.py・cellpose_track_and_visualize.pyと同じ構成で、
Zスライス間タグ付け・面積ノイズフィルタ・描画は共通関数を再利用し、
輪郭の作り方だけをdiffusion_flow.pyのモデル+to_contour_inputに差し替える。
"""

import argparse
import json

import torch

from main import (
    RESULTS_DIR, UNet,
    build_frame_split, compute_norm_bounds, compute_abs_bounds,
    pad_to_multiple, raw_path, mask_path,
)
from diffusion_flow import to_contour_input, DIFFUSION_FLOW_MODELS_DIR
from track_and_visualize import (
    track_contours_across_z, compute_tag_avg_areas, compute_gt_avg_areas,
    filter_tags_by_area, render_area_histogram, render_slice_image, render_3d_view,
    NOISE_AREA_THRESHOLD,
)

import numpy as np
import tifffile

OUTPUT_TITLE = "output (前景確率、flow場は輪郭抽出にのみ使用)"
OUTPUT_CMAP = "gray"
OUTPUT_VMAX = 1.0


def load_model(model_path, device):
    model = UNet(out_channels=3, activation="sigmoid_tanh2").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    model.eval()
    return model


def predict_slice(model, raw_slice, norm_lo, norm_hi, device, close_radius=0):
    image = np.clip((raw_slice.astype(np.float32) - norm_lo) / (norm_hi - norm_lo), 0, 1)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    tensor_p, (h, w) = pad_to_multiple(tensor)
    with torch.no_grad():
        pred = model(tensor_p)[:, :, :h, :w]
    return to_contour_input(pred, image=image, close_radius=close_radius)


def process_frame(model, frame_idx, abs_lo, abs_hi, norm_lo, norm_hi, device, out_dir, plotly_bundle_relpath,
                   close_radius=0):
    mask_volume = tifffile.imread(mask_path(frame_idx))
    nonzero = np.where(mask_volume.sum(axis=(1, 2)) > 0)[0]
    if len(nonzero) == 0:
        return None
    z_range = list(range(int(nonzero.min()), int(nonzero.max()) + 1))
    true_cell_ids = set(np.unique(mask_volume[z_range])) - {0}

    raw_slices, contours_per_slice, prob_maps = [], [], []
    for z in z_range:
        raw = tifffile.imread(raw_path(frame_idx), key=z)
        raw_slices.append(raw)
        contours, prob = predict_slice(model, raw, norm_lo, norm_hi, device, close_radius)
        contours_per_slice.append(contours)
        prob_maps.append(prob)

    per_slice_tagged, cumulative_tags, total_tags = track_contours_across_z(
        contours_per_slice, raw_slices[0].shape)

    frame_dir = out_dir / f"t{frame_idx:03d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    slices_meta = []
    for i, z in enumerate(z_range):
        fname = f"z{z:03d}.png"
        render_slice_image(raw_slices[i], mask_volume[z], prob_maps[i], per_slice_tagged[i],
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
    parser.add_argument("--model-path", default=None,
                         help="未指定ならmodels/<dataset>/diffusion_flow/unet_best.pthを使う")
    parser.add_argument("--close-radius", type=int, default=0,
                         help="前景マスクにモルフォロジー・クロージングをかける半径(px)。"
                              "0なら従来通り(かけない)。細胞内部の暗い模様で前景に穴が空き、"
                              "軌跡積分が1つの細胞を2つに分けてしまう問題への対策"
                              "(2026-07-16、t051で確認)。再学習不要、輪郭抽出の後処理のみ")
    parser.add_argument("--run-name", default=None,
                         help="未指定なら--close-radiusに応じて自動決定")
    parser.add_argument("--model-label", default="binary_DiffusionFlow")
    parser.add_argument("--contour-label", default=None,
                         help="未指定なら--close-radiusに応じて自動決定")
    parser.add_argument("--description", default=None)
    args = parser.parse_args()

    if args.close_radius > 0:
        run_name = args.run_name or "full_trained_diffusion_flow_closed"
        contour_label = args.contour_label or f"熱拡散flow場+前景クロージング(radius={args.close_radius})"
        description = args.description or (
            "binary_DiffusionFlowと同じモデル・同じ学習済み重みを再利用し、輪郭抽出の前段で"
            f"前景マスクにモルフォロジー・クロージング(半径{args.close_radius}px)をかけてから"
            "軌跡積分する。細胞内部の暗い模様で前景確率マップに穴が空き、その穴のせいで"
            "1つの細胞が2つのタグに分裂してしまう問題への対策(2026-07-16、t051で確認、"
            "PROJECT_DiffusionFlow.md参照)。再学習は行っていない。")
    else:
        run_name = args.run_name or "full_trained_diffusion_flow"
        contour_label = args.contour_label or "熱拡散flow場による軌跡積分"
        description = args.description or (
            "Fluo-N3DH-CHOのST(Silver Truth)で(前景,熱拡散ベースflow場dy/dx)の3chを学習"
            "(35epoch、Early Stoppingせず完走)。本物のGT(t033/t062)でCTC SEGスコア=0.689、"
            "同条件のCellpose(0.625)・DWT 1epoch(0.576)を上回った。PROJECT_DiffusionFlow.md参照。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path or (DIFFUSION_FLOW_MODELS_DIR / "unet_best.pth")
    model = load_model(model_path, device)

    train_frames, val_frames, test_frames = build_frame_split()
    norm_lo, norm_hi = compute_norm_bounds(train_frames)
    abs_lo, abs_hi = compute_abs_bounds(train_frames)

    import plotly.offline as pyo
    (RESULTS_DIR / "plotly_bundle.js").write_text(pyo.get_plotlyjs(), encoding="utf-8")

    out_dir = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_meta = []
    for f in test_frames:
        print(f"processing t{f:03d} ...")
        result = process_frame(model, f, abs_lo, abs_hi, norm_lo, norm_hi, device, out_dir, "../../plotly_bundle.js",
                                close_radius=args.close_radius)
        if result is not None:
            frames_meta.append(result)
            print(f"  GT={result['nGtTotal']} pred={result['nPredTotal']} "
                  f"pred_filtered={result['nPredFiltered']} slices={len(result['slices'])}")

    manifest = {
        "run_name": run_name, "description": description,
        "model_label": args.model_label, "contour_label": contour_label,
        "frames": frames_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (out_dir / "description.txt").write_text(description, encoding="utf-8")
    print(f"saved: {out_dir}/manifest.json ({len(frames_meta)} frames)")


if __name__ == "__main__":
    main()

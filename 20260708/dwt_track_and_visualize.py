"""
track_and_visualize.pyのDWT(dwt.py)版。既存のtrack_and_visualize.pyは単一の
LABEL_SPECモデルを前提にしているため(`load_model`が`UNet(out_channels=LABEL_SPEC...)`
を組み立てる、`predict_slice`が`LABEL_SPEC.to_contour_input`を呼ぶ)、DN+WTNの
2ネットワーク構成のDWTはそのままでは動かせない。Zスライス間のタグ付け・面積フィルタ・
描画(render_slice_image等)は汎用なのでtrack_and_visualize.pyから再利用し、
モデルの読み込みと推論(predict_slice相当)だけをDWT用に差し替える。

生成物の形式(manifest.json・PNG・view3d.html)はtrack_and_visualize.pyと同じなので、
build_viewer.pyは変更なしでそのままDWTのrunを取り込める。
"""

import argparse
import json
from pathlib import Path

import torch

from main import (
    RESULTS_DIR, UNet,
    build_frame_split, compute_norm_bounds, compute_abs_bounds,
    pad_to_multiple, raw_path, mask_path,
)
from dwt import DWT_N_BINS, DWT_CUT_LEVEL, DWT_MODELS_DIR, to_contour_input
from track_and_visualize import (
    track_contours_across_z, compute_tag_avg_areas, compute_gt_avg_areas,
    filter_tags_by_area, render_area_histogram, render_slice_image, render_3d_view,
    NOISE_AREA_THRESHOLD,
)

import numpy as np
import tifffile

OUTPUT_TITLE = f"output (DWT energy 0-{DWT_N_BINS - 1}, 0=bg/margin)"
OUTPUT_CMAP = "viridis"
OUTPUT_VMAX = DWT_N_BINS - 1


def load_dwt_models(dn_path, wtn_path, device):
    dn_model = UNet(in_channels=1, out_channels=2, activation="tanh").to(device)
    wtn_model = UNet(in_channels=3, out_channels=DWT_N_BINS, activation="none").to(device)
    dn_model.load_state_dict(torch.load(dn_path, map_location=device)["model"])
    wtn_model.load_state_dict(torch.load(wtn_path, map_location=device)["model"])
    dn_model.eval()
    wtn_model.eval()
    return dn_model, wtn_model


def load_dwt_models_from_joint(joint_path, device):
    """dwt.pyのStage3は、nn.ModuleDict({"dn":dn_model,"wtn":wtn_model})を1つの
    チェックポイントとして保存する(main.save_checkpoint経由)。state_dictのキーは
    "dn."/"wtn."で始まるので、プレフィックスを外してそれぞれのUNetに振り分ける。
    Stage3まで完走した後は、単体チェックポイントより結合fine-tuning後のこちらを使うべき
    (DNとWTNの整合性が実際に強制されているのはjoint_bestのみ、PROJECT_DWT.md参照)。"""
    dn_model = UNet(in_channels=1, out_channels=2, activation="tanh").to(device)
    wtn_model = UNet(in_channels=3, out_channels=DWT_N_BINS, activation="none").to(device)
    state_dict = torch.load(joint_path, map_location=device)["model"]
    dn_state = {k[len("dn."):]: v for k, v in state_dict.items() if k.startswith("dn.")}
    wtn_state = {k[len("wtn."):]: v for k, v in state_dict.items() if k.startswith("wtn.")}
    dn_model.load_state_dict(dn_state)
    wtn_model.load_state_dict(wtn_state)
    dn_model.eval()
    wtn_model.eval()
    return dn_model, wtn_model


def predict_slice_dwt(dn_model, wtn_model, raw_slice, norm_lo, norm_hi, device, cut_level):
    """(輪郭リスト, 表示用エネルギークラスmap)を返す。track_and_visualize.pyの
    predict_sliceに相当(そちらはLABEL_SPEC.to_contour_input、こちらはdwt.to_contour_input)。"""
    image = np.clip((raw_slice.astype(np.float32) - norm_lo) / (norm_hi - norm_lo), 0, 1)
    image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    return to_contour_input(dn_model, wtn_model, image_t, device, cut_level)


def process_frame(dn_model, wtn_model, frame_idx, abs_lo, abs_hi, norm_lo, norm_hi,
                   device, cut_level, out_dir, plotly_bundle_relpath):
    mask_volume = tifffile.imread(mask_path(frame_idx))
    nonzero = np.where(mask_volume.sum(axis=(1, 2)) > 0)[0]
    if len(nonzero) == 0:
        return None
    z_range = list(range(int(nonzero.min()), int(nonzero.max()) + 1))
    true_cell_ids = set(np.unique(mask_volume[z_range])) - {0}

    raw_slices, contours_per_slice, energy_maps = [], [], []
    for z in z_range:
        raw = tifffile.imread(raw_path(frame_idx), key=z)
        raw_slices.append(raw)
        contours, energy_map = predict_slice_dwt(dn_model, wtn_model, raw, norm_lo, norm_hi, device, cut_level)
        contours_per_slice.append(contours)
        energy_maps.append(energy_map)

    per_slice_tagged, cumulative_tags, total_tags = track_contours_across_z(
        contours_per_slice, raw_slices[0].shape)

    frame_dir = out_dir / f"t{frame_idx:03d}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    slices_meta = []
    for i, z in enumerate(z_range):
        fname = f"z{z:03d}.png"
        render_slice_image(raw_slices[i], mask_volume[z], energy_maps[i], per_slice_tagged[i],
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
    parser.add_argument("--joint-path", default=str(DWT_MODELS_DIR / "joint_best.pth"),
                         help="Stage3(結合fine-tuning)のチェックポイント。存在すればこちらを優先して使う")
    parser.add_argument("--dn-path", default=str(DWT_MODELS_DIR / "dn_best.pth"),
                         help="--joint-pathが無い場合のフォールバック(DN単体のチェックポイント)")
    parser.add_argument("--wtn-path", default=str(DWT_MODELS_DIR / "wtn_best.pth"),
                         help="--joint-pathが無い場合のフォールバック(WTN単体のチェックポイント)")
    parser.add_argument("--cut-level", type=int, default=DWT_CUT_LEVEL)
    parser.add_argument("--run-name", required=True, help="results/以下に作る実験フォルダ名")
    parser.add_argument("--description", required=True, help="この実験の説明(description.txtに書く)")
    parser.add_argument("--model-label", default="DeepWatershedTransform (DN+WTN)")
    parser.add_argument("--contour-label", default=f"DWTエネルギークラスのカット(cut_level={DWT_CUT_LEVEL})")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Path(args.joint_path).exists():
        print(f"joint checkpoint使用: {args.joint_path}")
        dn_model, wtn_model = load_dwt_models_from_joint(Path(args.joint_path), device)
    else:
        print(f"jointチェックポイント無し。単体チェックポイントを使用: {args.dn_path}, {args.wtn_path}")
        dn_model, wtn_model = load_dwt_models(Path(args.dn_path), Path(args.wtn_path), device)

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
        result = process_frame(dn_model, wtn_model, f, abs_lo, abs_hi, norm_lo, norm_hi,
                                device, args.cut_level, out_dir, "../../plotly_bundle.js")
        if result is not None:
            frames_meta.append(result)
            print(f"  GT={result['nGtTotal']} pred={result['nPredTotal']} "
                  f"pred_filtered={result['nPredFiltered']} slices={len(result['slices'])}")

    manifest = {
        "run_name": args.run_name, "description": args.description,
        "model_label": args.model_label, "contour_label": args.contour_label,
        "frames": frames_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (out_dir / "description.txt").write_text(args.description, encoding="utf-8")
    print(f"saved: {out_dir}/manifest.json ({len(frames_meta)} frames)")


if __name__ == "__main__":
    main()

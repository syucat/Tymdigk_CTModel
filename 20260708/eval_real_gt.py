"""
Fluo-N3DH-CHOの本物のGT/SEG(人手検証済み、`01_GT/SEG/man_seg_*.tif`)だけを使った評価。

このデータセットは普段`ST`(Silver Truth、アルゴリズム生成の疑似正解)を正解として
使っているが(main.py DATASET_CONFIGS参照)、STはそれ自体が別のアルゴリズムの出力で
あり、本当に正しいかは別問題(PROJECT_Cellpose.md参照)。本物のGTは19個の
(フレーム, Zスライス)にしか存在しない疎なデータだが、これを使えば「STとどれだけ
似ているか」ではなく「本物の正解とどれだけ一致するか」を確認できる。

**注意(公平性)**: 19枚のうちtrain/val/test分割でtestに入っているのは
t033・t062の2枚だけ。それ以外の17枚はこのプロジェクト自身が学習した
モデル(DWT、distance_transform等)にとってはtrain/valデータの可能性があり、
それらのモデルの評価に使うと不当に良い数字が出る(リーク)。一方、Cellposeは
このデータセットを一切学習していない(事前学習のみ)ので、19枚全部を
公平に使える。

評価指標: Cell Tracking Challenge(CTC)のSEG measureと同じ定義。
各GTインスタンスRについて、予測ラベルの中で最もRと重なる領域Lを探し、
|R∩L| > 0.5*|R| (過半数が重なる)ならJaccard指数|R∩L|/|R∪L|をRのスコアとし、
それ未満(対応する予測が無い)なら0点。全GTインスタンスの平均がSEGスコア。
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile

import main

REAL_GT_DIR = main.DATA_DIR / "01_GT" / "SEG"


def list_real_gt_slices():
    """(frame, z, path)のリストを返す。ファイル名は man_seg_{frame:03d}_{z:03d}.tif。"""
    slices = []
    for p in sorted(REAL_GT_DIR.glob("man_seg_*.tif")):
        parts = p.stem.split("_")  # ["man", "seg", "033", "001"]
        frame, z = int(parts[2]), int(parts[3])
        slices.append((frame, z, p))
    return slices


def seg_score(gt_label, pred_label):
    """CTCのSEG measureと同じ定義。(平均スコア, 一致した数, GTインスタンス総数)を返す。"""
    gt_ids = [i for i in np.unique(gt_label) if i != 0]
    if not gt_ids:
        return None, 0, 0
    scores = []
    matched = 0
    for gid in gt_ids:
        gt_bin = gt_label == gid
        gt_area = gt_bin.sum()
        overlapping_pred_ids = np.unique(pred_label[gt_bin])
        best_iou = 0.0
        for pid in overlapping_pred_ids:
            if pid == 0:
                continue
            pred_bin = pred_label == pid
            inter = (gt_bin & pred_bin).sum()
            if inter > 0.5 * gt_area:
                union = (gt_bin | pred_bin).sum()
                best_iou = max(best_iou, inter / union)
        scores.append(best_iou)
        if best_iou > 0:
            matched += 1
    return float(np.mean(scores)), matched, len(gt_ids)


def predict_cellpose(model, raw_slice, diameter=None):
    masks, _flows, _styles = model.eval(raw_slice, diameter=diameter)
    return masks


def predict_dwt(dn_model, wtn_model, device, cut_level, raw_slice, norm_lo, norm_hi):
    import torch
    from scipy import ndimage
    from skimage.segmentation import expand_labels

    image = np.clip((raw_slice.astype(np.float32) - norm_lo) / (norm_hi - norm_lo), 0, 1)
    image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
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
        return np.zeros_like(energy_class, dtype=np.int64)
    expanded = expand_labels(labels, distance=max(energy_class.shape))
    expanded[~fg_mask] = 0
    return expanded


def predict_diffusion_flow(model, device, raw_slice, norm_lo, norm_hi):
    import torch

    image = np.clip((raw_slice.astype(np.float32) - norm_lo) / (norm_hi - norm_lo), 0, 1)
    image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    image_p, (h, w) = main.pad_to_multiple(image_t)
    with torch.no_grad():
        pred = model(image_p)[:, :, :h, :w]
    fg_prob = pred[0, 0].cpu().numpy()
    flow_y = pred[0, 1].cpu().numpy()
    flow_x = pred[0, 2].cpu().numpy()
    fg_mask = fg_prob > 0.5
    return main._integrate_and_cluster(fg_mask, flow_y, flow_x)


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["cellpose", "dwt", "diffusion_flow"], required=True)
    parser.add_argument("--diffusion-flow-path", default=None,
                         help="--method diffusion_flow の時のチェックポイントパス。未指定なら"
                              "models/Fluo-N3DH-CHO/diffusion_flow/unet_best.pthを使う")
    parser.add_argument("--dwt-joint-path", default=None,
                         help="--method dwt の時のjointチェックポイントパス。未指定なら"
                              "models/Fluo-N3DH-CHO/dwt/joint_best.pthを使う")
    parser.add_argument("--cut-level", type=int, default=1)
    parser.add_argument("--all-19", action="store_true",
                         help="test分割外(train/valの可能性がある17枚)も含めて評価する。"
                              "Cellposeのような、このデータを学習していない手法でのみ公平")
    args = parser.parse_args()

    if main.DATASET_NAME != "Fluo-N3DH-CHO":
        print(f"警告: DATASET_NAME={main.DATASET_NAME}。本物のGT/SEGはFluo-N3DH-CHO用に"
              "書かれたパスなので、他データセットでは動作しない可能性があります。")

    _train, _val, test_frames = main.build_frame_split()
    test_frame_set = set(test_frames)

    slices = list_real_gt_slices()
    if not args.all_19:
        slices = [(f, z, p) for f, z, p in slices if f in test_frame_set]
        print(f"test分割内の本物GTのみ評価: {len(slices)}枚 "
              f"(train/valに含まれる可能性がある{19 - len(slices)}枚は除外。"
              "--all-19で全部使う)")
    else:
        n_test = sum(1 for f, z, p in slices if f in test_frame_set)
        print(f"全{len(slices)}枚を評価(うちtest分割内={n_test}枚、"
              f"train/valの可能性あり={len(slices) - n_test}枚)")

    if args.method == "cellpose":
        from cellpose import models
        model = models.CellposeModel(gpu=True)
        predict_fn = lambda raw: predict_cellpose(model, raw)
    elif args.method == "dwt":
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        joint_path = Path(args.dwt_joint_path) if args.dwt_joint_path else (
            main.Path(__file__).parent / "models" / main.DATASET_NAME / "dwt" / "joint_best.pth")
        from dwt_track_and_visualize import load_dwt_models_from_joint
        dn_model, wtn_model = load_dwt_models_from_joint(joint_path, device)
        train_frames, _val, _test = main.build_frame_split()
        norm_lo, norm_hi = main.compute_norm_bounds(train_frames)
        predict_fn = lambda raw: predict_dwt(dn_model, wtn_model, device, args.cut_level, raw, norm_lo, norm_hi)
    else:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = Path(args.diffusion_flow_path) if args.diffusion_flow_path else (
            main.Path(__file__).parent / "models" / main.DATASET_NAME / "diffusion_flow" / "unet_best.pth")
        model = main.UNet(out_channels=3, activation="sigmoid_tanh2").to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        model.eval()
        train_frames, _val, _test = main.build_frame_split()
        norm_lo, norm_hi = main.compute_norm_bounds(train_frames)
        predict_fn = lambda raw: predict_diffusion_flow(model, device, raw, norm_lo, norm_hi)

    results = []
    for frame, z, path in slices:
        gt_label = tifffile.imread(path)
        raw = tifffile.imread(main.raw_path(frame), key=z)
        pred_label = predict_fn(raw)
        score, matched, n_gt = seg_score(gt_label, pred_label)
        in_test = frame in test_frame_set
        results.append((frame, z, score, matched, n_gt, in_test))
        tag = "[test]" if in_test else "[train/val?]"
        print(f"t{frame:03d} z={z:03d} {tag}  SEG={score:.3f}  matched={matched}/{n_gt}")

    valid = [r for r in results if r[2] is not None]
    overall = float(np.mean([r[2] for r in valid]))
    test_only = [r for r in valid if r[5]]
    print(f"\n=== {args.method} 全体SEGスコア(平均) ===")
    print(f"全{len(valid)}枚: {overall:.4f}")
    if test_only:
        test_avg = float(np.mean([r[2] for r in test_only]))
        print(f"うちtest分割内{len(test_only)}枚のみ(公平な部分集合): {test_avg:.4f}")


if __name__ == "__main__":
    main_cli()

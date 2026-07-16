"""
指定したモデル(--model-path)でテストフレーム全15本のZスタックを予測し、
findContoursで輪郭(=タグ)を取り出してZスライス間で追跡する。

各Zスライスにつき、以下7パネルのPNGを保存する:
  base                        : そのスライス自身のmin-maxで自動調整した生データ
  input                       : 学習データ全体の絶対min-max(クリップ無し)で正規化したモノクロ
  input (brightness-modified) : パーセンタイルクリップ+min-max正規化（実際にモデルに入れる形）
  output (B:bg, W:cell)       : モデルのsigmoid確率マップ(0=背景,1=細胞、閾値なしのグラデーション)
  contour                     : 出力を二値化してfindContoursで取った輪郭(黒背景・白線のみ)
  tagged (C:IoU, Hungarian)    : 輪郭をZ間追跡してタグ付けした結果(黒背景・タグごとに色分け)
  GT mask (N)                 : GTの二値マスク、タイトルにそのスライスの細胞数Nを表示

さらにフレームごとに1つ、Zスライスによらず同じものを表示する「予測タグの3D vs GTの3D」の
回転可能なPlotlyビュー(view3d.html)を生成する。

結果はresults/<--run-name>/以下に、manifest.json・description.txtとして保存する。
複数runをまとめて見るビューアはbuild_viewer.pyで別途生成する。

トラッキングの考え方(コスト=1-IoU、ハンガリアン法で全体最適な1対1割り当て):
  ① 最初のZスライスは全輪郭に新規タグを発行
  ② 以降の各Zスライスで、前スライスの各タグ(track)と今スライスの各輪郭の
     コスト行列 cost[i][j] = 1 - IoU(track_i, contour_j) を作る
  ③ scipy.optimize.linear_sum_assignmentで合計コスト最小の1対1割り当てを求める
  ④ 割り当てのうちコストが閾値より高い(重なりが少なすぎる)ものは棄却する
  ⑤ 対応が付かなかった輪郭には新規タグを発行し、アクティブなタグ集合を更新する
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment

from main import (
    RESULTS_DIR, LABEL_SPEC, ANNOTATION_SUBDIR,
    build_frame_split, compute_norm_bounds, compute_abs_bounds,
    UNet, pad_to_multiple, raw_path, mask_path,
)

# "GT"(人手検証済み)か"ST"(Silver Truth、アルゴリズム生成の疑似正解)か。
# Fluo-N3DH-CHO/Fluo-C3DL-MDA231は本物のGT/SEGが疎すぎるためSTを使っている
# (main.py DATASET_CONFIGS参照)。パネルのタイトルにそのまま使い、「GT」と
# 表示していたのが実際にはSTだった、という混同を今後の生成分から防ぐ。
ANNOTATION_LABEL = ANNOTATION_SUBDIR

IOU_MATCH_THRESHOLD = 0.1
MIN_CONTOUR_AREA = 5
DARK_BG = "#1a1a1a"

# タグの「生涯平均面積」がこの値未満ならノイズ由来の偽輪郭とみなして除外する。
# 根拠(t137, full_trainedモデルでの実測): 本物らしい長命タグの平均面積=1211.7、
# 短命ノイズタグの平均面積=46.9〜55.1 と桁違いなので、150は両者をよく分離する。
# また「軌跡の最大到達サイズ/平均サイズで足切りする」こと自体、MOT分野やCTC関連の
# 細胞追跡研究(hard/soft area limitなど)で実際に使われている確立された手法
# (詳細はPROJECT_tagging.md参照)。
NOISE_AREA_THRESHOLD = 150


def load_model(model_path, device):
    model = UNet(out_channels=LABEL_SPEC.out_channels, activation=LABEL_SPEC.activation).to(device)
    ckpt = torch.load(model_path, map_location=device)
    # 新形式({"model": state_dict, "optimizer": ..., ...})・旧形式(state_dictそのもの)の両方に対応
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_slice(model, raw_slice, norm_lo, norm_hi, device):
    """(輪郭リスト, 表示用float map)を返す。実際の変換はLABEL_SPEC.to_contour_input
    (main.py)がLABEL_MODEごとに行う(単純な2値化+findContours、またはwatershedなど。
    詳細はPROJECT_labeling.md参照)。"""
    image = np.clip((raw_slice.astype(np.float32) - norm_lo) / (norm_hi - norm_lo), 0, 1)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    tensor_p, (h, w) = pad_to_multiple(tensor)
    with torch.no_grad():
        pred = model(tensor_p)[:, :, :h, :w]
    return LABEL_SPEC.to_contour_input(pred, image=image)


def contour_mask(contour, shape):
    m = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(m, [contour], -1, 1, thickness=cv2.FILLED)
    return m


def iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return inter / union if union > 0 else 0.0


def track_contours_across_z(contours_per_slice, shape):
    """
    Zスライスごとの輪郭リスト(predict_sliceで既に抽出済み)を受け取り、輪郭をZ間で
    コスト(1-IoU)ベースのハンガリアン法(scipy.optimize.linear_sum_assignment)で
    最適に1対1割り当てしてタグ(ID)を割り振る。戻り値: 各zごとの[(contour, tag_id), ...]、
    各zまでの累積ユニークタグ数のリスト、使われたタグの総数
    """
    active_tracks = {}  # tag_id -> 直前スライスでの輪郭マスク
    next_tag = 1
    per_slice_tagged = []
    cumulative_tags = []

    for contours in contours_per_slice:
        cur_masks = [contour_mask(c, shape) for c in contours]
        assigned = [None] * len(contours)

        if active_tracks and cur_masks:
            track_ids = list(active_tracks.keys())
            track_masks = [active_tracks[t] for t in track_ids]
            cost = np.ones((len(track_ids), len(cur_masks)))
            for i, tm in enumerate(track_masks):
                for j, cm in enumerate(cur_masks):
                    cost[i, j] = 1 - iou(tm, cm)

            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= 1 - IOU_MATCH_THRESHOLD:
                    assigned[c] = track_ids[r]

        for i in range(len(contours)):
            if assigned[i] is None:
                assigned[i] = next_tag
                next_tag += 1

        active_tracks = {assigned[i]: cur_masks[i] for i in range(len(contours))}
        per_slice_tagged.append(list(zip(contours, assigned)))
        cumulative_tags.append(next_tag - 1)

    return per_slice_tagged, cumulative_tags, next_tag - 1


def compute_tag_avg_areas(per_slice_tagged):
    """タグごとに、そのタグが登場した全スライスでの輪郭面積の平均(=生涯平均面積)を返す"""
    areas_by_tag = {}
    for slice_tagged in per_slice_tagged:
        for contour, tag_id in slice_tagged:
            areas_by_tag.setdefault(tag_id, []).append(cv2.contourArea(contour))
    return {tag_id: float(np.mean(areas)) for tag_id, areas in areas_by_tag.items()}


def compute_gt_avg_areas(mask_volume, z_range):
    """GTインスタンスIDごとに、登場した全スライスでのピクセル数の平均を返す"""
    areas_by_id = {}
    for z in z_range:
        ids, counts = np.unique(mask_volume[z], return_counts=True)
        for inst_id, count in zip(ids, counts):
            if inst_id == 0:
                continue
            areas_by_id.setdefault(int(inst_id), []).append(int(count))
    return {inst_id: float(np.mean(areas)) for inst_id, areas in areas_by_id.items()}


def filter_tags_by_area(per_slice_tagged, tag_areas, threshold):
    """タグの生涯平均面積がthreshold未満のものを除外したper_slice_taggedを返す"""
    keep_ids = {tag_id for tag_id, area in tag_areas.items() if area >= threshold}
    filtered = [[(c, t) for c, t in slice_tagged if t in keep_ids] for slice_tagged in per_slice_tagged]
    return filtered, keep_ids


def render_area_histogram(tag_areas_before, tag_areas_after, gt_areas, threshold, out_path):
    """
    タグの「生涯平均面積」の分布を、対数スケールのビン(1-2, 2-4, 4-8, ...)で
    集計した真のヒストグラム。予測タグとGTインスタンスは同じビンを共有するので
    1つのパネルに重ね描きして直接比較できる。上段=フィルタ前、下段=フィルタ後
    (閾値適用後の予測タグのみ)を並べて、フィルタの効果も一目で分かるようにする。
    """
    pred_before = list(tag_areas_before.values())
    pred_after = list(tag_areas_after.values())
    gt_vals = list(gt_areas.values())
    max_v = max(pred_before + gt_vals + [threshold, 2.0])
    n_bins = int(np.ceil(np.log2(max_v))) + 1
    edges = np.array([2.0 ** i for i in range(n_bins + 1)])  # 1,2,4,8,...
    log_edges = np.log2(edges)
    gt_counts, _ = np.histogram(gt_vals, bins=edges)

    # 予測とGTの棒を完全に重ねず、各ビン内で左右にずらして半分だけ重なるように描く
    # (対数軸上でも見た目のズレ幅が揃うよう、ログ空間で25%/75%の位置を計算する)
    def sub_span(frac_start, frac_end):
        lo = 2.0 ** (log_edges[:-1] + frac_start * (log_edges[1:] - log_edges[:-1]))
        hi = 2.0 ** (log_edges[:-1] + frac_end * (log_edges[1:] - log_edges[:-1]))
        return lo, hi

    pred_lo, pred_hi = sub_span(0.0, 0.75)
    gt_lo, gt_hi = sub_span(0.25, 1.0)

    bin_centers = 2.0 ** ((log_edges[:-1] + log_edges[1:]) / 2)
    bin_labels = [f"$2^{{{int(log_edges[i])}}}$~$2^{{{int(log_edges[i + 1])}}}$" for i in range(n_bins)]

    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    fig.patch.set_facecolor(DARK_BG)

    def draw(ax, pred_vals, label, show_xlabels):
        ax.set_facecolor(DARK_BG)
        pred_counts, _ = np.histogram(pred_vals, bins=edges)
        ax.bar(pred_lo, pred_counts, width=pred_hi - pred_lo, align="edge",
               color="#4a7dff", alpha=0.85, label=f"predicted tags (n={len(pred_vals)})")
        ax.bar(gt_lo, gt_counts, width=gt_hi - gt_lo, align="edge",
               color="#ff9f40", alpha=0.85, label=f"{ANNOTATION_LABEL} instances (n={len(gt_vals)})")
        ax.axvline(threshold, color="#ff4d4d", linestyle="--", linewidth=1.2,
                   label=f"noise threshold={threshold}")
        ax.set_ylabel("count", color="white", fontsize=9)
        ax.set_title(label, color="white", fontsize=10, loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(bin_centers)
        ax.set_xticklabels(bin_labels if show_xlabels else [])
        ax.minorticks_off()
        ax.tick_params(colors="white", labelsize=7)
        if show_xlabels:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for spine in ax.spines.values():
            spine.set_color("#666")

    draw(axes[0], pred_before, "before filtering", show_xlabels=False)
    draw(axes[1], pred_after, f"after area filter (threshold={threshold})", show_xlabels=True)

    axes[1].set_xlabel("lifetime avg area (px)", color="white", fontsize=9)

    legend = axes[0].legend(fontsize=8.5, facecolor="#242424", edgecolor="#555", loc="upper left")
    for text in legend.get_texts():
        text.set_color("white")

    fig.suptitle("lifetime avg area per tag: predicted vs GT (before / after noise filtering)",
                 color="white", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def tab20_color_mpl(tag_id):
    return tuple(int(c) for c in (np.array(plt.cm.tab20(tag_id % 20)[:3]) * 255))


def tab20_color_rgb_str(tag_id):
    r, g, b = plt.cm.tab20(tag_id % 20)[:3]
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def render_slice_image(raw_slice, gt_mask_slice, prob_map, tagged_contours,
                        abs_lo, abs_hi, norm_lo, norm_hi, out_path,
                        output_title=None, output_cmap=None, output_vmax=None):
    """output_title/cmap/vmaxを省略するとLABEL_SPEC(main.pyのLABEL_MODE)の値を使う
    (既存呼び出しとの後方互換)。DWTのようにLABEL_SPECを持たないパイプラインは
    ここを明示的に渡す(dwt_track_and_visualize.py参照)。"""
    output_title = LABEL_SPEC.output_title if output_title is None else output_title
    output_cmap = LABEL_SPEC.output_cmap if output_cmap is None else output_cmap
    output_vmax = LABEL_SPEC.output_vmax if output_vmax is None else output_vmax
    shape = raw_slice.shape
    input_mono = np.clip((raw_slice.astype(np.float32) - abs_lo) / (abs_hi - abs_lo), 0, 1)
    input_bmod = np.clip((raw_slice.astype(np.float32) - norm_lo) / (norm_hi - norm_lo), 0, 1)
    n_gt = len(set(np.unique(gt_mask_slice)) - {0})

    contour_img = np.zeros((*shape, 3), dtype=np.uint8)
    tagged_img = np.zeros((*shape, 3), dtype=np.uint8)
    for contour, tag_id in tagged_contours:
        cv2.drawContours(contour_img, [contour], -1, (255, 255, 255), 1)
        color = tab20_color_mpl(tag_id)
        cv2.drawContours(tagged_img, [contour], -1, color, 2)
        m = cv2.moments(contour)
        if m["m00"] > 0:
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            cv2.putText(tagged_img, str(tag_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # GTマスク自体が持つインスタンスIDごとに輪郭を色分け+番号表示する(予測のtagged_imgと同じ描き方)。
    # くっついて見える細胞が、実はGT上では別インスタンスとして区別されているのかどうかを
    # 目で確認できるようにするための表示専用パネル(トラッキング処理には影響しない)。
    gt_tagged_img = np.zeros((*shape, 3), dtype=np.uint8)
    for inst_id in np.unique(gt_mask_slice):
        if inst_id == 0:
            continue
        inst_id = int(inst_id)
        binary = (gt_mask_slice == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tab20_color_mpl(inst_id)
        for c in contours:
            cv2.drawContours(gt_tagged_img, [c], -1, color, 2)
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                cv2.putText(gt_tagged_img, str(inst_id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    fig.patch.set_facecolor(DARK_BG)

    panels = [
        (axes[0, 0], raw_slice, "base", dict(cmap="gray")),
        (axes[0, 1], input_mono, "input", dict(cmap="gray", vmin=0, vmax=1)),
        (axes[0, 2], input_bmod, "input (brightness-modified)", dict(cmap="gray", vmin=0, vmax=1)),
        (axes[0, 3], prob_map, output_title,
         dict(cmap=output_cmap, vmin=0, vmax=output_vmax)),
        (axes[1, 0], contour_img, "contour", {}),
        (axes[1, 1], tagged_img, "tagged (C:IoU, Hungarian)", {}),
        (axes[1, 2], (gt_mask_slice > 0).astype(np.uint8), f"{ANNOTATION_LABEL} mask ({n_gt})", dict(cmap="gray")),
        (axes[1, 3], gt_tagged_img, f"{ANNOTATION_LABEL} mask tagged by instance ID ({n_gt})", {}),
    ]
    for ax, data, title, kwargs in panels:
        im = ax.imshow(data, **kwargs)
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")
        ax.set_facecolor(DARK_BG)
        if ax is axes[0, 3]:
            # 「output」パネルだけ、値→色の対応が分かるカラーバーを添える
            # (distance_transformモードではpx単位の深さ、他モードでは0-1の確率)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.get_yticklabels(), color="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_3d_view(z_range, mask_volume, per_slice_tagged, out_path, plotly_bundle_relpath,
                    pred_title="predicted tags (3D, C:IoU, Hungarian)"):
    pred_by_tag = {}
    for i, z in enumerate(z_range):
        for contour, tag_id in per_slice_tagged[i]:
            pts = contour.reshape(-1, 2)
            pred_by_tag.setdefault(tag_id, []).append((pts[:, 0], pts[:, 1], z))

    gt_by_id = {}
    for z in z_range:
        slice_labels = mask_volume[z]
        for inst_id in np.unique(slice_labels):
            if inst_id == 0:
                continue
            binary = (slice_labels == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                pts = c.reshape(-1, 2)
                gt_by_id.setdefault(int(inst_id), []).append((pts[:, 0], pts[:, 1], z))

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                         subplot_titles=(pred_title, f"{ANNOTATION_LABEL} instances (3D)"))

    def add_traces(by_tag, col):
        for tag_id, segs in by_tag.items():
            xs, ys, zs = [], [], []
            for x_arr, y_arr, z in segs:
                xs.extend(x_arr.tolist() + [None])
                ys.extend(y_arr.tolist() + [None])
                zs.extend([z] * len(x_arr) + [None])
            fig.add_trace(
                go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                             line=dict(color=tab20_color_rgb_str(tag_id), width=3),
                             showlegend=False),
                row=1, col=col,
            )

    add_traces(pred_by_tag, 1)
    add_traces(gt_by_id, 2)

    fig.update_layout(template="plotly_dark", height=560, margin=dict(l=0, r=0, t=40, b=0))

    # 左右の3Dシーンで、片方を回転/ズームしたらもう片方にも同じカメラ位置を反映する。
    # Plotly.relayoutは非同期(Promiseを返す)なので、syncingの解除は完了後(.then)まで待つこと。
    # 同期的に解除すると、relayoutが引き起こす次のplotly_relayoutイベントと競合し、
    # 無限ループでタブがフリーズするバグになる。
    camera_sync_js = """
    var gd = document.getElementById('{plot_id}');
    var syncing = false;
    gd.on('plotly_relayout', function(eventdata) {
      if (syncing) return;
      var cam = eventdata['scene.camera'] || eventdata['scene2.camera'];
      if (!cam) return;
      var target = eventdata['scene.camera'] ? 'scene2.camera' : 'scene.camera';
      syncing = true;
      var update = {};
      update[target] = cam;
      Plotly.relayout(gd, update).then(function() {
        syncing = false;
      });
    });
    """
    fig.write_html(str(out_path), include_plotlyjs=plotly_bundle_relpath, full_html=True,
                   post_script=camera_sync_js)


def process_frame(model, frame_idx, abs_lo, abs_hi, norm_lo, norm_hi, device, out_dir, plotly_bundle_relpath):
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
        contours, prob = predict_slice(model, raw, norm_lo, norm_hi, device)
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
                            abs_lo, abs_hi, norm_lo, norm_hi, frame_dir / fname)
        n_gt_slice = len(set(np.unique(mask_volume[z])) - {0})
        slices_meta.append({
            "z": z, "file": f"t{frame_idx:03d}/{fname}",
            "nGt": n_gt_slice, "nPred": len(per_slice_tagged[i]), "cum": cumulative_tags[i],
        })

    render_3d_view(z_range, mask_volume, per_slice_tagged, frame_dir / "view3d.html", plotly_bundle_relpath)

    # 面積によるノイズタグ除去(閾値=NOISE_AREA_THRESHOLD、タグの生涯平均面積で判定)
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
    parser.add_argument("--model-path", required=True, help="読み込むモデルのパス(.pth)")
    parser.add_argument("--run-name", required=True, help="results/以下に作る実験フォルダ名")
    parser.add_argument("--description", required=True, help="この実験の説明(description.txtに書く)")
    parser.add_argument("--model-label", required=True,
                         help="ビューアの1段階目(学習/モデル)の表示名。同じLABEL_MODEでも"
                              "チェックポイントが違えば別モデル扱いになるため、自動導出はせず毎回指定する")
    parser.add_argument("--contour-label", required=True,
                         help="ビューアの2段階目(輪郭の作り方)の表示名(例: watershed, distance_flowなど)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.model_path), device)

    train_frames, val_frames, test_frames = build_frame_split()
    norm_lo, norm_hi = compute_norm_bounds(train_frames)
    abs_lo, abs_hi = compute_abs_bounds(train_frames)

    # 3D可視化用のplotly.jsを1つだけresults/直下に置き、各frameのview3d.htmlから相対参照する
    (RESULTS_DIR / "plotly_bundle.js").write_text(pyo.get_plotlyjs(), encoding="utf-8")

    out_dir = RESULTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_meta = []
    for f in test_frames:
        print(f"processing t{f:03d} ...")
        result = process_frame(model, f, abs_lo, abs_hi, norm_lo, norm_hi, device, out_dir, "../../plotly_bundle.js")
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

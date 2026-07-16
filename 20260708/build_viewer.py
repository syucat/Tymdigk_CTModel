"""
results/<dataset_name>/<run_name>/manifest.json を全データセット分まとめて、
results/result.html に1つの統合ビューアを生成する。

選択UIは「①モデル(学習済みチェックポイント)→②輪郭の作り方」の2段階になっている
(③のZスライス間タグ付けは現状コスト(1-IoU)+ハンガリアン法の1種類しかないため、
選択式にはせず固定表示)。②にはまだ`track_and_visualize.py`を実行していない
組み合わせ(`KNOWN_PLACEHOLDERS`参照)も「未生成」として表示され、選ぶと
生成用のコピペコマンドが出る。

track_and_visualize.pyを実行した後にこれを実行すると、ビューアに反映される。
"""

import importlib
import json
import os
import time
from pathlib import Path

RESULTS_ROOT = Path(__file__).parent / "results"

# まだtrack_and_visualize.pyを実行していないが、既存のLABEL_MODEのコードだけで
# 生成可能な組み合わせ。(dataset, model_label, contour_label, 生成コマンド)。
# 対応するrun_nameのmanifest.jsonが既に存在する場合は、自動的にこの一覧から除外される
# (実行後にここを手で消さなくてよい)。
KNOWN_PLACEHOLDERS = [
    dict(
        dataset="Fluo-N3DH-CHO",
        model_label="distance_transform(instance-max正規化)",
        contour_label="distance_flow(後付け微分)",
        run_name="full_trained_distance_flow",
        command=(
            'LABEL_MODE=distance_flow DATASET_NAME=Fluo-N3DH-CHO python 20260708/track_and_visualize.py '
            '--model-path 20260708/models/Fluo-N3DH-CHO/distance_transform/unet_best.pth '
            '--run-name full_trained_distance_flow '
            '--model-label "distance_transform(instance-max正規化)" '
            '--contour-label "distance_flow(後付け微分)" '
            '--description "distance_transformモデル(instance-max正規化)に対し、watershedの代わりに'
            '深さの勾配から導いた方向ベクトル場で軌跡積分+クラスタリングする(設計A、後付け微分)。'
            't005のGT4/11は分離できるが、孤立細胞の過剰分割が起きる(輪郭数656→1091、PROJECT_labeling2.md参照)。"'
        ),
    ),
    dict(
        dataset="Fluo-N3DH-CHO",
        model_label="distance_flow_trained(4ch学習済み方向)",
        contour_label="学習済み方向による軌跡積分",
        run_name="full_trained_distance_flow_trained",
        command=(
            'LABEL_MODE=distance_flow_trained DATASET_NAME=Fluo-N3DH-CHO python 20260708/track_and_visualize.py '
            '--model-path 20260708/models/Fluo-N3DH-CHO/distance_flow_trained/unet_best.pth '
            '--run-name full_trained_distance_flow_trained '
            '--model-label "distance_flow_trained(4ch学習済み方向)" '
            '--contour-label "学習済み方向による軌跡積分" '
            '--description "(前景,深さ,dx,dy)の4チャンネルを最初から学習させたモデル。過剰分割は解消するが'
            '(輪郭数653)、t005のGT4/11分離は失われmerged割合も悪化する(1.53%)。'
            '111エポックでEarly Stopping、test IoU=0.7901(PROJECT_labeling2.md参照)。"'
        ),
    ),
]

# ②「輪郭の作り方」の各手法についての簡潔な技術説明(runごとの個別description
# ではなく、手法そのものの一般的な説明。contour_labelの値をキーにする)。
CONTOUR_METHOD_INFO = {
    "単純二値化(findContours)": "前景確率を0.5で2値化し、そのままfindContoursで輪郭を取る。接触した細胞の分離は行わない。",
    "内部クラス二値化(findContours)": "背景/内部/境界の3クラスのうち、内部クラスだけを2値化してfindContoursにかける。"
                                      "境界が壁になり接触細胞を分離できる想定だったが、効果はほぼ無かった(PROJECT_labeling.md参照)。",
    "watershed": "深さ(境界までの距離)の地形の頂点を種にしてwatershedで領域分割する。t005のGT4/11の"
                 "ような接触ペアは分離できない(PROJECT_labeling.md参照)。",
    "distance_flow(後付け微分)": "深さチャンネルの予測値を数値微分して方向ベクトルを求め、軌跡積分+クラスタリングで"
                                  "分離する(設計A)。GT4/11の分離はできるが、予測ノイズが増幅され孤立細胞の過剰分割が"
                                  "起きる(PROJECT_labeling2.md参照)。",
    "学習済み方向による軌跡積分": "(前景,深さ,dx,dy)の4chをGTから直接学習し、モデル自身が出す方向で軌跡積分する。"
                                  "過剰分割は解消するが、接触ペアの分離効果も同時に失われる(PROJECT_labeling2.md参照)。",
    "単純二値化(findContours) + 実データ": "前景の判定はbinaryモデルのまま(再学習なし)、輪郭抽出だけを変更。"
                                            "前景マスクの中で、元の入力画像の明るさ(ガウシアンぼかしsigma=20で"
                                            "平滑化)を地形としてwatershedにかける。マスクの形だけでは持ち得ない、"
                                            "実際の輝度データを分離の手がかりに使う試み。t005のGT4/11が正しく"
                                            "分離できているかはまだ未検証。",
    "DWTエネルギークラスのカット(cut_level=1)": "Deep Watershed Transform(Bai & Urtasun, CVPR 2017)。"
                                            "Direction Network(方向ベクトル)とWatershed Transform Network"
                                            "(K段階の離散エネルギークラス、境界付近を重み付き損失で重点学習)の"
                                            "2ネットワークを3段階(DN単体→WTN単体→結合fine-tuning)で学習し、"
                                            "推論時は固定の1つの閾値でエネルギークラスをカットして連結成分を"
                                            "取るだけで輪郭を分離する(peak_local_maxのようなピーク数の決定が"
                                            "不要)。詳細はPROJECT_DWT.md参照。",
}


def get_split_info(dataset_name):
    """指定データセットのtrain/val/test内訳を返す。main.pyはDATASET_NAMEを
    起動時の環境変数から読むモジュール定数として持つので、データセットを切り替えて
    再利用するにはos.environを書き換えた上でモジュールをreloadする必要がある。"""
    os.environ["DATASET_NAME"] = dataset_name
    import main
    importlib.reload(main)
    train_frames, val_frames, test_frames = main.build_frame_split()
    return {
        "train": len(train_frames),
        "val": len(val_frames),
        "test": len(test_frames),
        "testFrames": [f"t{f:03d}" for f in test_frames],
        # "GT"(人手検証済み)か"ST"(Silver Truth、アルゴリズム生成の疑似正解)か。
        # Fluo-N3DH-CHO/Fluo-C3DL-MDA231はGT/SEGが疎すぎるためSTを使っている
        # (main.py DATASET_CONFIGS参照)。ビューア上で「GT」と表示していたのは
        # 実際にはSTだったため、annotationTypeで動的に正しい表記に切り替える。
        "annotationType": main.ANNOTATION_SUBDIR,
    }


def main():
    runs = []
    split_info = {}
    existing_run_keys = set()
    for manifest_path in sorted(RESULTS_ROOT.glob("*/*/manifest.json")):
        dataset_name = manifest_path.parent.parent.name
        run = json.loads(manifest_path.read_text(encoding="utf-8"))
        run["dataset"] = dataset_name
        runs.append(run)
        existing_run_keys.add((dataset_name, run["run_name"]))
        if dataset_name not in split_info:
            split_info[dataset_name] = get_split_info(dataset_name)

    if not runs:
        print("results/*/*/manifest.json が見つかりません。先にtrack_and_visualize.pyを実行してください。")
        return

    # 既に生成済み(manifest.jsonが存在する)ものはplaceholder一覧から除外する
    placeholders = [
        p for p in KNOWN_PLACEHOLDERS
        if (p["dataset"], p["run_name"]) not in existing_run_keys
    ]

    manifest_json = json.dumps(runs, ensure_ascii=False)
    placeholders_json = json.dumps(placeholders, ensure_ascii=False)
    split_info_json = json.dumps(split_info, ensure_ascii=False)
    contour_info_json = json.dumps(CONTOUR_METHOD_INFO, ensure_ascii=False)
    build_version = int(time.time())

    html = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>細胞トラッキング結果ビューア</title>
<style>
  body {{ font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 12px; }}
  .layout {{ display: flex; gap: 12px; align-items: flex-start; }}
  .panel {{ background: #242424; border: 1px solid #3a3a3a; border-radius: 8px; padding: 10px; }}
  #run-panel {{ width: 460px; flex-shrink: 0; display: flex; gap: 8px; }}
  #model-col {{ width: 220px; flex-shrink: 0; }}
  #contour-col {{ flex: 1; min-width: 0; }}
  #view-panel {{ flex: 1; min-width: 0; text-align: center; }}
  .col-title {{ font-size: 11px; color: #777; margin-bottom: 4px; }}
  .dataset-header {{
    font-size: 11.5px; font-weight: bold; color: #888; text-transform: uppercase;
    margin: 12px 0 4px; letter-spacing: 0.03em;
  }}
  .dataset-header:first-child {{ margin-top: 0; }}
  .model-header {{
    font-size: 12.5px; font-weight: bold; color: #ccc; margin: 8px 0 4px; padding-left: 2px;
  }}
  .stage-item, .contour-item {{
    border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px 8px; margin-bottom: 6px;
    cursor: pointer; font-size: 12.5px; margin-left: 6px;
  }}
  .stage-item:hover, .contour-item:hover {{ border-color: #4a7dff; }}
  .stage-item.active, .contour-item.active {{ border-color: #4a7dff; background: #2c3550; }}
  .contour-item.placeholder-item {{ border-style: dashed; color: #999; }}
  .contour-item.placeholder-item.active {{ border-color: #ffb066; background: #3a2f20; color: #eee; }}
  .run-desc {{ font-size: 11.5px; color: #9db8ff; margin-top: 4px; }}
  .toggle-arrow {{ display: inline-block; width: 14px; color: #777; cursor: pointer; }}
  .toggle-arrow:hover {{ color: #4a7dff; }}
  img {{ max-width: 100%; border: 1px solid #444; background: #1a1a1a; }}
  #summary {{ font-size: 13px; margin-bottom: 8px; }}
  #tag-method-info {{ font-size: 12px; color: #999; margin-bottom: 10px; text-align: left; }}
  .slider-row {{ display: flex; align-items: center; gap: 10px; margin: 10px 0; }}
  .slider-row label {{ font-size: 13px; width: 76px; flex-shrink: 0; text-align: right; }}
  .slider-row input[type=range] {{ flex: 1; }}
  .slider-val {{ font-size: 12.5px; color: #bbb; min-width: 260px; text-align: left; }}
  #view3d, #view3dFiltered {{ width: 100%; height: 560px; border: 1px solid #444; background: #1a1a1a; margin-top: 14px; }}
  h2.section {{ font-size: 14px; text-align: left; margin: 18px 0 4px; color: #ccc; }}
  .compare-box {{
    text-align: left; font-size: 13px; background: #2c2c2c; border: 1px solid #444;
    border-radius: 6px; padding: 8px 12px; margin: 6px 0 10px;
  }}
  .compare-box b {{ color: #ffb066; }}
  .hist-row {{ display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }}
  .hist-row img {{ max-width: 48%; }}
  #split-info {{
    font-size: 12.5px; color: #bbb; background: #242424; border: 1px solid #3a3a3a;
    border-radius: 6px; padding: 8px 12px; margin-bottom: 12px; line-height: 1.6;
  }}
  #placeholder-box {{ text-align: left; padding: 20px 0; }}
  .command-box {{
    background: #1a1a1a; border: 1px solid #444; border-radius: 4px; padding: 8px;
    font-family: monospace; font-size: 11.5px; white-space: pre-wrap; word-break: break-all;
    margin-top: 8px; color: #b6ffb0;
  }}
  .copy-btn {{
    margin-top: 8px; font-size: 12px; padding: 4px 10px; cursor: pointer;
    background: #2c3550; color: #eee; border: 1px solid #4a7dff; border-radius: 4px;
  }}
</style>
</head>
<body>
<h1>細胞セグメンテーション &rarr; findContours &rarr; Zスライス間トラッキング 結果ビューア</h1>
<div id="split-info"></div>
<div class="layout">
  <div class="panel" id="run-panel">
    <div id="model-col"><div class="col-title">① モデル</div></div>
    <div id="contour-col"><div class="col-title">② 輪郭の作り方</div></div>
  </div>
  <div class="panel" id="view-panel">
    <div id="tag-method-info">③ タグ付け手法: コスト(1-IoU)+ハンガリアン法(全run共通)</div>
    <div id="placeholder-box" style="display:none;">
      <p>まだ生成されていません。次のコマンドを<code>OAR_practice/OAR_practice</code>直下で実行してください:</p>
      <div class="command-box" id="command-text"></div>
      <button class="copy-btn" onclick="copyCommand()">コピー</button>
    </div>
    <div id="result-content">
      <div id="summary"></div>
      <img id="viewer" src="">
      <div class="slider-row">
        <label for="zSlider">Z</label>
        <input type="range" id="zSlider" min="0" value="0" oninput="onZSlide(this.value)">
        <span class="slider-val" id="zVal"></span>
      </div>
      <div class="slider-row">
        <label for="frameSlider">フレーム</label>
        <input type="range" id="frameSlider" min="0" value="0" oninput="onFrameSlide(this.value)">
        <span class="slider-val" id="frameVal"></span>
      </div>

      <div class="compare-box" id="filterCompare"></div>

      <h2 class="section">タグ別 生涯平均面積のヒストグラム(フレーム全体、Zに依らず同じもの。予測とGTを重ね描き、上段=フィルタ前/下段=フィルタ後)</h2>
      <img id="areaHist" src="">

      <h2 class="section">3D再構成(Zスライスによらず同じもの。ドラッグで回転できます。フィルタ前)</h2>
      <iframe id="view3d" src=""></iframe>

      <h2 class="section">3D再構成(面積閾値でノイズタグ除去後)</h2>
      <iframe id="view3dFiltered" src=""></iframe>
    </div>
  </div>
</div>
<script>
const RUNS = {manifest_json};
const PLACEHOLDERS = {placeholders_json};
const SPLIT_INFO = {split_info_json};
const CONTOUR_METHOD_INFO = {contour_info_json};
const BUILD_VERSION = {build_version};  // 再生成のたびに変わるので、ブラウザキャッシュを回避できる

let selectedDataset = null, selectedModel = null, selectedStage = null;
let runIdx = null, placeholderIdx = null;
let expandedStages = new Set();   // "dataset|model|stage" 単位で、説明文を開いているものを覚えておく(基本は畳んだ状態)
let expandedContours = new Set(); // contour_label単位
let frameIdx = 0, zIdx = 0;

function datasetsInOrder() {{
  const seen = [];
  RUNS.forEach(r => {{ if (!seen.includes(r.dataset)) seen.push(r.dataset); }});
  PLACEHOLDERS.forEach(p => {{ if (!seen.includes(p.dataset)) seen.push(p.dataset); }});
  return seen;
}}

function modelsForDataset(dataset) {{
  const seen = [];
  RUNS.forEach(r => {{ if (r.dataset === dataset && !seen.includes(r.model_label)) seen.push(r.model_label); }});
  PLACEHOLDERS.forEach(p => {{ if (p.dataset === dataset && !seen.includes(p.model_label)) seen.push(p.model_label); }});
  return seen;
}}

// run_nameの命名規則(先頭が"1epoch"か"full_trained"か)から学習段階を判定する。
// 学習段階は「輪郭の作り方」とは独立した、①モデル側の軸(どれだけ学習したか)。
function stageOf(runName) {{
  return runName.startsWith('1epoch') ? '1epoch' : 'full_trained';
}}

function stagesForModel(dataset, model) {{
  const seen = [];
  RUNS.forEach(r => {{
    if (r.dataset === dataset && r.model_label === model) {{
      const s = stageOf(r.run_name);
      if (!seen.includes(s)) seen.push(s);
    }}
  }});
  PLACEHOLDERS.forEach(p => {{
    if (p.dataset === dataset && p.model_label === model) {{
      const s = stageOf(p.run_name);
      if (!seen.includes(s)) seen.push(s);
    }}
  }});
  return seen;
}}

// そのモデル・学習段階の代表的な説明文(既存runがあればその説明を使う。1epoch/full_trained
// それぞれにつき通常1つのrunしか無いので、これで一意に決まる)
function stageDescription(dataset, model, stage) {{
  const r = RUNS.find(r => r.dataset === dataset && r.model_label === model && stageOf(r.run_name) === stage);
  return r ? r.description : '';
}}

function itemsForStage(dataset, model, stage) {{
  const items = [];
  RUNS.forEach((r, i) => {{
    if (r.dataset === dataset && r.model_label === model && stageOf(r.run_name) === stage) {{
      items.push({{type: 'run', idx: i, label: r.contour_label}});
    }}
  }});
  PLACEHOLDERS.forEach((p, i) => {{
    if (p.dataset === dataset && p.model_label === model && stageOf(p.run_name) === stage) {{
      items.push({{type: 'placeholder', idx: i, label: p.contour_label}});
    }}
  }});
  return items;
}}

function esc(s) {{ return encodeURIComponent(s); }}

function toggleStage(event, key) {{
  event.stopPropagation();
  if (expandedStages.has(key)) expandedStages.delete(key); else expandedStages.add(key);
  renderModelCol();
}}

function toggleContour(event, label) {{
  event.stopPropagation();
  if (expandedContours.has(label)) expandedContours.delete(label); else expandedContours.add(label);
  renderContourCol();
}}

function renderModelCol() {{
  let html = '<div class="col-title">① モデル(学習段階)</div>';
  datasetsInOrder().forEach(dataset => {{
    html += `<div class="dataset-header">${{dataset}}</div>`;
    modelsForDataset(dataset).forEach(model => {{
      html += `<div class="model-header">${{model}}</div>`;
      stagesForModel(dataset, model).forEach(stage => {{
        const active = (dataset === selectedDataset && model === selectedModel && stage === selectedStage) ? 'active' : '';
        const key = dataset + '|' + model + '|' + stage;
        const expanded = expandedStages.has(key);
        const arrow = expanded ? '▼' : '▶';
        html += `<div class="stage-item ${{active}}" ` +
                `onclick="selectStage(decodeURIComponent('${{esc(dataset)}}'), decodeURIComponent('${{esc(model)}}'), decodeURIComponent('${{esc(stage)}}'))">` +
                `<span class="toggle-arrow" onclick="toggleStage(event, decodeURIComponent('${{esc(key)}}'))">${{arrow}}</span>${{stage}}` +
                (expanded ? `<div class="run-desc">${{stageDescription(dataset, model, stage)}}</div>` : '') +
                `</div>`;
      }});
    }});
  }});
  document.getElementById('model-col').innerHTML = html;
}}

function renderContourCol() {{
  let html = '<div class="col-title">② 輪郭の作り方</div>';
  if (selectedStage) {{
    itemsForStage(selectedDataset, selectedModel, selectedStage).forEach(item => {{
      const expanded = expandedContours.has(item.label);
      const arrow = expanded ? '▼' : '▶';
      const info = CONTOUR_METHOD_INFO[item.label] || '';
      const descHtml = expanded ? `<div class="run-desc">${{info}}</div>` : '';
      const toggleHtml = `<span class="toggle-arrow" onclick="toggleContour(event, decodeURIComponent('${{esc(item.label)}}'))">${{arrow}}</span>`;
      if (item.type === 'run') {{
        const active = (placeholderIdx === null && runIdx === item.idx) ? 'active' : '';
        html += `<div class="contour-item ${{active}}" onclick="selectRun(${{item.idx}})">${{toggleHtml}}${{item.label}}${{descHtml}}</div>`;
      }} else {{
        const active = (placeholderIdx === item.idx) ? 'active' : '';
        html += `<div class="contour-item placeholder-item ${{active}}" onclick="selectPlaceholder(${{item.idx}})">${{toggleHtml}}${{item.label}} (未生成)${{descHtml}}</div>`;
      }}
    }});
  }}
  document.getElementById('contour-col').innerHTML = html;
}}

function annotationLabel(dataset) {{
  return SPLIT_INFO[dataset].annotationType;
}}

function annotationNote(dataset) {{
  const t = annotationLabel(dataset);
  return t === 'GT'
    ? '正解データ: <b>GT</b>(人手検証済みの正解、Fluo-N3DH-SIM+はシミュレーション生成なので厳密に既知)'
    : '正解データ: <b>ST</b>(Silver Truth。人手検証済みのGTではなく、複数アルゴリズムの結果から作った疑似正解。'
      + '本物のGT/SEGは疎で、このデータセットではごく一部のフレーム・Zスライスにしか存在しない)';
}}

function renderSplitInfo() {{
  const info = SPLIT_INFO[selectedDataset];
  document.getElementById('split-info').innerHTML =
    `データセット: <b>${{selectedDataset}}</b> ／ ` +
    `学習(train): ${{info.train}}フレーム ／ ` +
    `検証(val, Early Stoppingの判定用): ${{info.val}}フレーム ／ ` +
    `<b>テスト(test, このビューアで表示中): ${{info.test}}フレーム</b> ` +
    `(${{info.testFrames.join(', ')}})<br>` +
    `${{annotationNote(selectedDataset)}}<br>` +
    "※ 表示しているのは学習に使っていないtestフレームの予測結果です。以下のパネルの" +
    "「GT mask」等の表記は、このデータセットが実際にはSTの場合でも旧表記のまま残っています。";
}}

function setupSliders() {{
  const frames = RUNS[runIdx].frames;
  const frameSlider = document.getElementById('frameSlider');
  frameSlider.max = frames.length - 1;
  frameSlider.value = frameIdx;

  const zSlider = document.getElementById('zSlider');
  zSlider.max = frames[frameIdx].slices.length - 1;
  zSlider.value = zIdx;
}}

// フレーム(またはrun)が変わった時だけ呼ぶ: サマリーと3Dビュー・ヒストグラムを更新する
function renderFrameLevel() {{
  const frame = RUNS[runIdx].frames[frameIdx];
  const runPrefix = RUNS[runIdx].dataset + "/" + RUNS[runIdx].run_name + "/";
  const label = annotationLabel(RUNS[runIdx].dataset);
  document.getElementById('summary').textContent =
    `t${{String(frame.frame).padStart(3, '0')}}　${{label}}の真の3D細胞数: ${{frame.nGtTotal}} 個 ／ 予測の最終ユニークタグ数: ${{frame.nPredTotal}} 個`;
  document.getElementById('frameVal').textContent =
    `t${{String(frame.frame).padStart(3, '0')}} [${{frameIdx + 1}}/${{RUNS[runIdx].frames.length}}]　${{label}}:${{frame.nGtTotal}} / 予測:${{frame.nPredTotal}}`;
  document.getElementById('view3d').src = runPrefix + frame.view3d + "?v=" + BUILD_VERSION;

  if (frame.areaHist) {{
    document.getElementById('areaHist').src = runPrefix + frame.areaHist + "?v=" + BUILD_VERSION;
  }}
  if (frame.view3dFiltered) {{
    document.getElementById('view3dFiltered').src = runPrefix + frame.view3dFiltered + "?v=" + BUILD_VERSION;
  }}
  if (frame.nPredFiltered !== undefined) {{
    const removed = frame.nPredTotal - frame.nPredFiltered;
    document.getElementById('filterCompare').innerHTML =
      `${{label}}: <b>${{frame.nGtTotal}}</b> 個　／　フィルタ前 予測: <b>${{frame.nPredTotal}}</b> 個　` +
      `&rarr;　フィルタ後 予測: <b>${{frame.nPredFiltered}}</b> 個` +
      `（面積閾値=${{frame.noiseAreaThreshold}}pxで ${{removed}} 個のタグをノイズとして除外）`;
  }}
}}

// Zスライダーが変わった時に呼ぶ: 2D画像とZラベルだけ更新する(3Dは再読み込みしない)
function renderZLevel() {{
  const frame = RUNS[runIdx].frames[frameIdx];
  const s = frame.slices[zIdx];
  const runPrefix = RUNS[runIdx].dataset + "/" + RUNS[runIdx].run_name + "/";
  const label = annotationLabel(RUNS[runIdx].dataset);
  document.getElementById('viewer').src = runPrefix + s.file + "?v=" + BUILD_VERSION;
  document.getElementById('zVal').textContent =
    `z=${{s.z}} [${{zIdx + 1}}/${{frame.slices.length}}]　このスライスの${{label}}細胞数: ${{s.nGt}}　予測検出数: ${{s.nPred}}　累積タグ数: ${{s.cum}}`;
}}

function selectStage(dataset, model, stage) {{
  selectedDataset = dataset; selectedModel = model; selectedStage = stage;
  runIdx = null; placeholderIdx = null;
  const items = itemsForStage(dataset, model, stage);
  if (items.length > 0) {{
    if (items[0].type === 'run') {{ selectRun(items[0].idx); return; }}
    else {{ selectPlaceholder(items[0].idx); return; }}
  }}
  renderModelCol(); renderContourCol();
}}

function selectPlaceholder(i) {{
  placeholderIdx = i; runIdx = null;
  const p = PLACEHOLDERS[i];
  selectedDataset = p.dataset; selectedModel = p.model_label; selectedStage = stageOf(p.run_name);
  document.getElementById('result-content').style.display = 'none';
  document.getElementById('placeholder-box').style.display = 'block';
  document.getElementById('command-text').textContent = p.command;
  renderSplitInfo();
  renderModelCol(); renderContourCol();
}}

function copyCommand() {{
  const text = document.getElementById('command-text').textContent;
  navigator.clipboard.writeText(text);
}}

function selectRun(i) {{
  runIdx = i; placeholderIdx = null; frameIdx = 0; zIdx = 0;
  selectedDataset = RUNS[i].dataset; selectedModel = RUNS[i].model_label; selectedStage = stageOf(RUNS[i].run_name);
  document.getElementById('placeholder-box').style.display = 'none';
  document.getElementById('result-content').style.display = 'block';
  renderModelCol(); renderContourCol(); renderSplitInfo(); setupSliders(); renderFrameLevel(); renderZLevel();
}}

function onFrameSlide(v) {{
  frameIdx = parseInt(v, 10); zIdx = 0;
  setupSliders(); renderFrameLevel(); renderZLevel();
}}

function onZSlide(v) {{
  zIdx = parseInt(v, 10);
  renderZLevel();
}}

// 初期表示: 最初に見つかったデータセット・モデル・学習段階を選ぶ
(function init() {{
  const firstDataset = datasetsInOrder()[0];
  const firstModel = modelsForDataset(firstDataset)[0];
  const firstStage = stagesForModel(firstDataset, firstModel)[0];
  selectStage(firstDataset, firstModel, firstStage);
}})();
</script>
</body>
</html>
"""
    out_path = RESULTS_ROOT / "result.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"saved: {out_path} ({len(runs)} runs, {len(placeholders)} placeholders, across {len(split_info)} datasets)")


if __name__ == "__main__":
    main()

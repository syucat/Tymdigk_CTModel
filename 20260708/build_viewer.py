"""
results/<run_name>/manifest.json を全部集めて、results/result.htmlビューアを生成する。
track_and_visualize.pyを実行した後にこれを実行すると、ビューアに反映される。
"""

import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    runs = []
    for manifest_path in sorted(RESULTS_DIR.glob("*/manifest.json")):
        runs.append(json.loads(manifest_path.read_text(encoding="utf-8")))

    if not runs:
        print("results/*/manifest.json が見つかりません。先にtrack_and_visualize.pyを実行してください。")
        return

    manifest_json = json.dumps(runs, ensure_ascii=False)
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
  #run-panel {{ width: 260px; flex-shrink: 0; }}
  #view-panel {{ flex: 1; min-width: 0; text-align: center; }}
  .run-item {{
    border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px 8px; margin-bottom: 6px;
    cursor: pointer; font-size: 13px;
  }}
  .run-item:hover {{ border-color: #4a7dff; }}
  .run-item.active {{ border-color: #4a7dff; background: #2c3550; }}
  .run-desc {{ font-size: 11.5px; color: #9db8ff; margin-top: 2px; }}
  img {{ max-width: 100%; border: 1px solid #444; background: #1a1a1a; }}
  #summary {{ font-size: 13px; margin-bottom: 8px; }}
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
</style>
</head>
<body>
<h1>細胞セグメンテーション &rarr; findContours &rarr; Zスライス間トラッキング 結果ビューア</h1>
<div class="layout">
  <div class="panel" id="run-panel"></div>
  <div class="panel" id="view-panel">
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
<script>
const RUNS = {manifest_json};
const BUILD_VERSION = {build_version};  // 再生成のたびに変わるので、ブラウザキャッシュを回避できる
let runIdx = 0, frameIdx = 0, zIdx = 0;

function renderRuns() {{
  document.getElementById('run-panel').innerHTML = RUNS.map((r, i) => `
    <div class="run-item ${{i === runIdx ? 'active' : ''}}" onclick="selectRun(${{i}})">
      <div>${{r.run_name}}</div>
      <div class="run-desc">${{r.description}}</div>
    </div>`).join('');
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
  const runPrefix = RUNS[runIdx].run_name + "/";
  document.getElementById('summary').textContent =
    `t${{String(frame.frame).padStart(3, '0')}}　GTの真の3D細胞数: ${{frame.nGtTotal}} 個 ／ 予測の最終ユニークタグ数: ${{frame.nPredTotal}} 個`;
  document.getElementById('frameVal').textContent =
    `t${{String(frame.frame).padStart(3, '0')}} [${{frameIdx + 1}}/${{RUNS[runIdx].frames.length}}]　GT:${{frame.nGtTotal}} / 予測:${{frame.nPredTotal}}`;
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
      `GT: <b>${{frame.nGtTotal}}</b> 個　／　フィルタ前 予測: <b>${{frame.nPredTotal}}</b> 個　` +
      `&rarr;　フィルタ後 予測: <b>${{frame.nPredFiltered}}</b> 個` +
      `（面積閾値=${{frame.noiseAreaThreshold}}pxで ${{removed}} 個のタグをノイズとして除外）`;
  }}
}}

// Zスライダーが変わった時に呼ぶ: 2D画像とZラベルだけ更新する(3Dは再読み込みしない)
function renderZLevel() {{
  const frame = RUNS[runIdx].frames[frameIdx];
  const s = frame.slices[zIdx];
  document.getElementById('viewer').src = RUNS[runIdx].run_name + "/" + s.file + "?v=" + BUILD_VERSION;
  document.getElementById('zVal').textContent =
    `z=${{s.z}} [${{zIdx + 1}}/${{frame.slices.length}}]　このスライスのGT細胞数: ${{s.nGt}}　予測検出数: ${{s.nPred}}　累積タグ数: ${{s.cum}}`;
}}

function selectRun(i) {{
  runIdx = i; frameIdx = 0; zIdx = 0;
  renderRuns(); setupSliders(); renderFrameLevel(); renderZLevel();
}}

function onFrameSlide(v) {{
  frameIdx = parseInt(v, 10); zIdx = 0;
  setupSliders(); renderFrameLevel(); renderZLevel();
}}

function onZSlide(v) {{
  zIdx = parseInt(v, 10);
  renderZLevel();
}}

renderRuns();
setupSliders();
renderFrameLevel();
renderZLevel();
</script>
</body>
</html>
"""
    out_path = RESULTS_DIR / "result.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"saved: {out_path} ({len(runs)} runs)")


if __name__ == "__main__":
    main()

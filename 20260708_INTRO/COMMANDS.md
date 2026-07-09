# 20260708/ 実行コマンド集(配布用)

これは `20260708/`(細胞セグメンテーションU-Net + トラッキング実験)を
初めて受け取った人向けに、「何がしたいか」ごとに実行すべきコマンドをまとめたものです。
詳しい背景・アルゴリズムの説明は `20260708/PROJECT.md` を参照してください
(このファイルはコマンド操作だけに絞っています)。

---

## 0. フォルダの置き方(最初に確認)

配布物を展開したら、次のように **`20260708/` と `DATA_CELL/` を同じ親フォルダの直下**
に置いてください(重要: `DATA_CELL/` を `20260708/` の中にネストしないこと)。

```
どこかの親フォルダ/
├─ 20260708/
├─ 20260708_INTRO/   ← このCOMMANDS.mdとrequirements.txt
└─ DATA_CELL/        ← データセット(下記1.2で用意)
```

`20260708/main.py` の中で `DATA_DIR = 親フォルダ/DATA_CELL/Fluo-N3DH-SIM+` という
相対パスでデータを探しにいくため、配置がずれると全てのコマンドがエラーになります。

コマンドは基本的に **`20260708/` フォルダの中に `cd` してから** 実行してください。

**すでに入っているもの**: `20260708/models/unet_best.pth`(学習済みモデル)と
`20260708/results/`(生成済みの可視化結果、`result.html`含む)は配布物に含まれています。
なので「結果を見るだけ」であれば、下記1.のセットアップの一部(Pythonやvenvは不要)で
すぐ確認できます。

---

## 1. セットアップ(最初に1回だけ)

### 1.1 Python環境の構築

Python 3.10系がインストールされている前提です。

```bash
cd 20260708_INTRO
python3 -m venv venv
source venv/bin/activate        # Windowsの場合: venv\Scripts\activate
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**注意**:
- NVIDIA GPU(CUDA)を持っていないPCの場合は、`torch`のインストールコマンドから
  `--index-url ...`を外してCPU版を入れてください。学習(`main.py`)はCPUでも動きますが
  非常に遅くなります(GPUで6時間強かかった学習が、CPUだと現実的な時間で終わらない可能性大)。
- venvは `20260708/` の外(`20260708_INTRO/`など)に作るのがおすすめです。
  `20260708/`直下に作ると、後述の「フォルダをそのままZIPで再配布」する際にvenvの
  巨大なファイル群まで巻き込んでしまいます。

### 1.2 データセットの用意(学習・再推論をする場合のみ必要)

「結果を見るだけ」の人はここは飛ばして良いです。

`20260708/main.py` が使うのは `Fluo-N3DH-SIM+` のみです
(`DATA_CELL/`には`BF-C2DL-HSC`という別データセットも入っていますが、
20260708の実験では使わないのでダウンロード不要です)。

```bash
wget -c http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-SIM+.zip -O DATA_CELL/Fluo-N3DH-SIM+.zip
unzip DATA_CELL/Fluo-N3DH-SIM+.zip -d DATA_CELL/
```

**注意**: 約3.6GBあります。詳細・出典は `DATA_CELL/howToGet.md` 参照。

---

## 2. やりたいこと別コマンド

### A. 完成済みの結果を見るだけ(学習・データセット不要)

`20260708/results/result.html` をブラウザで直接開くだけです。

```bash
cd 20260708/results
python3 -m http.server 8000
# ブラウザで http://localhost:8000/result.html を開く
```

**注意**: `result.html`をブラウザで直接ダブルクリック(`file://`)で開くと、
回転可能な3Dビュー(Plotly)が読み込めず真っ白になることがあります。
その場合は上記のように`http.server`で配信してから開いてください。
run(実験)/フレーム/Zスライスの3段階を切り替えて見られます。

---

### B. 学習済みモデルで新しく可視化・追跡をやり直したい(再学習なし)

必要なもの: 1.1のPython環境 + 1.2のデータセット。既存の`models/unet_best.pth`
(または `models/`内の他の.pth)をそのまま使います。

```bash
cd 20260708
python3 track_and_visualize.py \
  --model-path models/unet_best.pth \
  --run-name my_experiment \
  --description "何を試したrunか一言で"
python3 build_viewer.py
```

- `track_and_visualize.py`: テスト15フレーム分のZスタックを予測→輪郭抽出→
  Zスライス間追跡(ハンガリアン法)まで行い、`results/my_experiment/`に
  フレームごとの7パネルPNG・3Dビュー・`manifest.json`を書き出します。
  **時間がかかります**(GPUで全フレーム分の推論+画像生成をするため、環境によっては
  数十分単位)。
- `--run-name`は**既存のフォルダ名(`1epoch`, `full_trained`など)と被らないもの**
  にしてください。同名にすると`results/<run-name>/`の中身が上書きされます。
- `build_viewer.py`は引数不要。`track_and_visualize.py`を実行した**後に必ず**
  実行しないと、新しいrunが`result.html`に反映されません。

---

### C. モデルを最初から(または続きから)学習し直したい

必要なもの: 1.1のPython環境 + 1.2のデータセット。GPU推奨(CPUだと非現実的な時間)。

```bash
cd 20260708
python3 main.py --quick   # まず数分で動作確認(3フレーム・3エポックだけ)
```

動作確認できたら本番の学習:

```bash
python3 main.py --batch-size 4 --max-vram-gb 8 --max-gpu-util 0.8
```

- `--max-vram-gb`: このプロセスが使うVRAM上限(GB)。お使いのGPUのVRAM容量に合わせて
  下げてください(元の実験はVRAM 12GBのGPUで上限8GBに設定していました)。
- `--max-gpu-util`: 平均GPU使用率の目標(0〜1)。発熱・電力が気になる場合に使います。
  無指定でも動きますが、フル速度でGPUを回し続けます。
- **重要な注意**: `models/unet_best.pth`が既に存在する場合、`main.py`は
  **自動的にそこから学習を再開します**(ゼロから学習し直しません)。
  アーキテクチャを変えた等の理由で完全にゼロからやり直したい場合は、
  事前に`models/unet_best.pth`を削除するかリネームしてから実行してください。
  逆に言えば、Ctrl-Cで止めても`models/unet_best.pth`さえ残っていれば
  次回起動時に続きから再開できます。
- 学習が終わったら、B.の`track_and_visualize.py` → `build_viewer.py`を
  実行すると新しいモデルの結果を`result.html`で確認できます。

#### (任意)GPU温度の安全監視

長時間学習中にGPU温度が心配な場合、**別のターミナル**で以下を並行して動かすと、
GPU温度が80℃を超えたら学習プロセスを自動一時停止(`SIGSTOP`)し、
75℃まで下がったら自動再開(`SIGCONT`)します(学習の進捗は失われません)。

```bash
bash 20260708/gpu_thermal_guard.sh
```

**注意**: `nvidia-smi`コマンドが使える環境(NVIDIA GPU + ドライバ導入済み)でのみ
動作します。プロセス名`python3 main.py`をpgrepで探す仕組みなので、
`main.py`を`python`など別名で起動していると検知できません。

---

### D. (上級者向け/メンテナンス用)結果の一部だけ再生成したい

B.の`track_and_visualize.py`は7パネルPNGの生成に時間がかかるため、
「3Dビューだけ」「面積フィルタ関連だけ」を直したいときに使う補助スクリプトです。
**対象の`run-name`が既に`track_and_visualize.py`で一度生成済みであること**が前提です。

```bash
cd 20260708
# 3Dビュー(view3d.html)だけ再生成。2DパネルPNGは再利用する
python3 regen_3d_only.py --model-path models/unet_best.pth --run-name full_trained

# 面積ヒストグラム + 面積フィルタ後3Dビューだけ追加生成し、manifest.jsonに追記する
python3 regen_area_and_filtered_3d.py --model-path models/unet_best.pth --run-name full_trained
```

**注意**: `regen_area_and_filtered_3d.py`は既存の`results/<run-name>/manifest.json`を
読み込んで書き換えます。実行前に、必要であれば`manifest.json`をバックアップしてください。
普段の利用でこの2つを使う場面は基本的にありません(通常はB.のフルコースで十分です)。

---

## 3. コマンド早見表

| やりたいこと | コマンド | 事前に必要なもの |
|---|---|---|
| 結果を見るだけ | `python3 -m http.server` (results/内で) | なし |
| 既存モデルで再可視化 | `track_and_visualize.py` → `build_viewer.py` | Python環境 + データセット |
| 学習を試す(動作確認) | `main.py --quick` | Python環境 + データセット |
| 本番学習 | `main.py --batch-size 4 --max-vram-gb N --max-gpu-util 0.8` | Python環境 + データセット + GPU推奨 |
| GPU温度監視(任意) | `gpu_thermal_guard.sh` (別ターミナル) | nvidia-smiが使える環境 |
| 3Dビューだけ再生成 | `regen_3d_only.py` | 生成済みrunが存在すること |
| 面積フィルタだけ再生成 | `regen_area_and_filtered_3d.py` | 生成済みrunが存在すること |

---

## 4. 配布媒体について

`20260708/`単体で約720MB(内訳: 学習済みモデル約150MB、生成済み結果約570MB)あり、
一般的なメール添付の上限(多くの大学メールは25MB前後)を大きく超えます。
また、GitHubなどのgitリポジトリ経由での共有もおすすめしません
(このリポジトリの`.gitignore`で`*.pth`や`results/`が意図的に除外されているのも
サイズが理由で、gitはそもそもこの手の大きい生成物の配布には向いていません)。

おすすめは次のいずれかです。

- **Google DriveやOneDrive、大学のBox等のクラウドストレージに`20260708/`を
  ZIPでアップロードし、共有リンクを渡す**方法が一番手軽です。
- 同じ研究室・同じネットワーク内であれば、共有ネットワークドライブやUSBメモリでの
  受け渡しでも十分です(むしろアップロード/ダウンロードの時間を省けます)。

`DATA_CELL/`(データセット本体)は**送らずに**、`20260708_INTRO/`と`20260708/`だけを
渡し、受け取った学生には本COMMANDS.mdの1.2の手順で各自ダウンロードしてもらうことを
おすすめします。理由:
- Fluo-N3DH-SIM+だけで3.6GBあり、送るものがさらに倍近く重くなる。
- Cell Tracking Challengeの公開データセットなので、各自が同じURLから取得可能。
- 「結果を見るだけ」の学生はそもそもデータセット自体が不要(2.A参照)。

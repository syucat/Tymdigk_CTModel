# DATA_CELL — データセット入手方法

このディレクトリの中身(各データセットのフォルダ)はサイズが大きいため
`.gitignore`で除外し、git管理には含めていません。このファイルだけを残して
おくので、必要な時は以下の手順で再取得してください。

すべて出典は Cell Tracking Challenge (http://celltrackingchallenge.net/) で、
ダウンロードURLは`http://data.celltrackingchallenge.net/training-datasets/<名前>.zip`
という共通の命名規則。

## Fluo-N3DH-SIM+

- ダウンロードURL: http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-SIM+.zip
- サイズ: 約3.6GB (展開後、01/ と 02/ の各シーケンス + ERR_SEG/GT)
- 取得スクリプト: `20260620/m_real.py`(`ensure_dataset()` 関数がwgetでダウンロード→展開まで自動化)
  - `DATASETS_DIR`は`DATA_CELL/`を指すので、スクリプトを再実行すればこのディレクトリに展開される
- 手動取得する場合:
  ```bash
  wget -c http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-SIM+.zip -O DATA_CELL/Fluo-N3DH-SIM+.zip
  unzip DATA_CELL/Fluo-N3DH-SIM+.zip -d DATA_CELL/
  ```

**細胞の説明**: 細胞核(nuclei)、**シミュレーションで生成**されたデータ(名前の"SIM+"の通り、
実写ではなくCGで作られた合成データ)。3D+時間(1フレーム=Zスタック、全150フレームに
密なGTあり)。形状は丸〜楕円のシンプルな塊。シミュレーションなのでノイズが少なく
学習しやすい(実測: テストIoU=0.857、56epochでEarly Stopping)。**挑戦度は低め**
(実データではない、形状も単純)だが、パイプラインの土台として最初に検証するには適していた。

## BF-C2DL-HSC

- ダウンロードURL: http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip
- サイズ: 約2.1GB (01/, 02/ の各シーケンス + ERR_SEG/GT/ST)
- 手動取得する場合:
  ```bash
  wget -c http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip -O DATA_CELL/BF-C2DL-HSC.zip
  unzip DATA_CELL/BF-C2DL-HSC.zip -d DATA_CELL/
  ```

**細胞の説明**: 造血幹細胞(Hematopoietic Stem Cell)、明視野(Brightfield)撮影の**実データ**。
**2D+時間**(Fluo-N3DH-SIM+と違ってZスタックが無く、1フレーム=1枚の2D画像。1010×1010、
1764フレームのうち手動GT(`SEG`)があるのは49枚だけの疎なアノテーション)。
**前景(細胞)ピクセルの割合は実測平均0.17%**と極端に少なく、通常のBCE+Dice損失のままだと
モデルが「全部背景」と答える状態に収束してしまい**学習が実質失敗する**ことを確認済み
(2026-07-09、ランダム初期値を変えて2回再現、両方とも同じ収束点)。重み付き損失
(pos_weightやFocal Loss相当)などの対策をしない限り、現状の実装では使えない。

## Fluo-C3DL-MDA231

- ダウンロードURL: http://data.celltrackingchallenge.net/training-datasets/Fluo-C3DL-MDA231.zip
- サイズ: 約184MB (01/, 02/ の各シーケンス + ERR_SEG/GT/ST)
- 手動取得する場合:
  ```bash
  wget -c http://data.celltrackingchallenge.net/training-datasets/Fluo-C3DL-MDA231.zip -O DATA_CELL/Fluo-C3DL-MDA231.zip
  unzip DATA_CELL/Fluo-C3DL-MDA231.zip -d DATA_CELL/
  ```

**細胞の説明**: 乳がん細胞(MDA-MB-231、浸潤性の高い間葉系細胞株)、細胞核だけでなく
**細胞全体(細胞質ごと)**をセグメンテーションする**実データ**。3D+時間(1フレーム=Zスタック、
01シーケンス12フレーム)。**偽足を伸ばした紡錘形・枝分かれした不定形**で、
Fluo-N3DH-SIM+/Fluo-N3DH-CHOの「丸い核」とは形状カテゴリが根本的に異なる(実際の
Zmax投影画像で目視確認済み)。**挑戦度は高い**(形が複雑、前景ピクセル比率も実測平均0.66%と
不均衡がやや強い)。
**注意**: 手動GT(`<seq>_GT/SEG/`)は`man_seg_<frame>_<z>.tif`という命名で、1フレームにつき
数枚のZスライスだけの疎なアノテーション(Fluo-N3DH-SIM+のように全フレーム・全Z密ではない)。
学習には代わりに`<seq>_ST/SEG/man_seg<frame>.tif`(Silver Truth、アルゴリズム生成だが
全フレーム・全Z密でFluo-N3DH-SIM+と同じ形式)を使う。`main.py`の`DATASET_CONFIGS`で
`annotation_subdir="ST"`として設定済み。

## Fluo-N3DH-CHO

- ダウンロードURL: http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CHO.zip
- サイズ: 約104MB (01/, 02/ の各シーケンス + ERR_SEG/GT/ST)
- 手動取得する場合:
  ```bash
  wget -c http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-CHO.zip -O DATA_CELL/Fluo-N3DH-CHO.zip
  unzip DATA_CELL/Fluo-N3DH-CHO.zip -d DATA_CELL/
  ```

**細胞の説明**: CHO細胞(チャイニーズハムスター卵巣細胞)の**核**、**実データ**。3D+時間
(1フレーム=Zスタック、01シーケンス92フレーム)。**形状はFluo-N3DH-SIM+と同じ「丸〜楕円の核」
カテゴリ**(Zmax投影画像で目視確認済み。核同士がくっついて写っている箇所はあるが、
トポロジー自体は単純)なので、**形状の新規性という意味では挑戦度は低い**。一方で
**前景ピクセルの割合は実測平均14%**とBF-C2DL-HSC(0.17%)より遥かにバランスが良く、
「実データでもパイプラインが正しく学習できるか」を確認する動作確認用データセットとして適する。
Fluo-C3DL-MDA231と同様、手動GT(`SEG`)は疎なので、学習には`ST`(Silver Truth)を使う
(`main.py`の`DATASET_CONFIGS`で設定済み)。

## 経緯

- 2026-06-20: `20260620/m_real.py`がFluo-N3DH-SIM+を`20260620/datasets/`にダウンロード
- 2026-06-25: `20260625/`の実験でFluo-N3DH-SIM+(同一内容)とBF-C2DL-HSCを`20260625/datasets/`に再取得
- 2026-07-02: 重複していたFluo-N3DH-SIM+を`diff -rq`で完全一致確認の上、`DATA_CELL/`に一本化。
  `20260620/datasets/`(重複分)は削除し、参照コードのパスを`DATA_CELL/`に更新
- 2026-07-09: `20260708/main.py`をデータセット横断対応に拡張(`DATASET_CONFIGS`で
  is_3d/frame_digits/annotation_subdirを切替)。BF-C2DL-HSCで学習を試したが、
  クラス不均衡による収束失敗を確認
- 2026-07-10: Fluo-C3DL-MDA231・Fluo-N3DH-CHOを追加取得。両データセットの手動GT/SEGが
  疎な命名規則(`man_seg_<frame>_<z>.tif`)であることを発見し、全フレーム密なST/SEGを
  使うよう対応

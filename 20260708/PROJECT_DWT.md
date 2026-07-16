# Deep Watershed Transform (DWT) の導入検討 (2026-07-15〜)

## 背景

既存のwatershed系手法(distance_transformモードの`peak_local_max`+watershed)は、
「本物の接触境界かノイズによる偶然の谷か」を区別する仕組みを持たないという構造的欠陥が
あった([PROJECT_labeling3.md](PROJECT_labeling3.md)セクション1)。4チャンネル設計
(前景,距離,dx,dy)も、dx,dyが独立損失で学習されるため、モデル自身が出す深さの微分と
整合する保証がない、という疑問が出ていた(同セクション3)。

Deep Watershed Transform (Bai & Urtasun, CVPR 2017, [arXiv:1611.08303](https://arxiv.org/abs/1611.08303))
は、この2つの問題に対応する設計を持つ既存研究として採用を検討した。

## 手法の要点

2つのネットワークからなる:

- **Direction Network (DN)**: 各画素の「境界から遠ざかる単位方向ベクトル」(2ch)を予測。
  角度誤差の二乗を、1/sqrt(インスタンス面積)で重み付けした損失で学習。
- **Watershed Transform Network (WTN)**: 画像+方向ベクトルを入力に、K段階(論文はK=16)に
  離散化した「エネルギークラス」を予測。ビン0=背景 または 境界から2px以内、
  ビン1〜K-1=インスタンス内部(距離が大きいほど高いビン)。重み付きクロスエントロピー損失
  (低エネルギー帯の誤りを重視)。

学習は3段階: ①DN単体を事前学習 → ②WTNを、GTの方向ベクトルを入力として単体で
事前学習 → ③DN+WTNを結合し、DNの予測方向を実際にWTNに渡してend-to-endでfine-tuning。
③で初めてDNとWTNの整合性が構造的に強制される(既存の4ch設計にはこれが無かった)。

推論時は、WTNの出力(エネルギークラス)をカットレベル(固定の1つの閾値)で閾値処理し、
残った連結成分をそのまま輪郭とする。`peak_local_max`のようなピーク数の決定が不要で、
単一の固定閾値で接触細胞が分離される設計になっている。

## このプロジェクトでの適応(論文との違い)

- 論文は複数の意味クラス(人・車…)ごとにカットレベルを変えるが、ここは「細胞」1クラス
  のみなのでカットレベルは1つの定数(`DWT_CUT_LEVEL`)。
- エネルギービンの境界(論文は「学習データ全体でピクセル数のバランスを取る」としか
  書かれておらず正確な式が不明)は、「境界からmargin px以内=ビン0」以外の前景ピクセルの
  インスタンスごとmax正規化深さ(既存のdistance_transformモードのdepthと同じ値)を、
  学習データ全体でプールしてから等分位点(quantile)でK-2分割する、という実装で代用。
- 損失の重み付け(論文の正確な式ckが不明)も、クラスごとのピクセル数の逆数の平方根で代用。

## 実装

`dwt.py`に新規実装。既存の`main.py`のインフラ(データセット分割・正規化・`UNet`クラス・
`pad_to_multiple`・`get_contours`・チェックポイント保存/読み込み)を`import main`で再利用し、
DN/WTN固有の部分(教師データ構築・損失・3段階学習ループ・推論時の輪郭構築)だけを追加した。

`UNet`クラスに`in_channels`パラメータ(WTNは画像+方向で3ch入力)と`activation="tanh"`
(DNの2ch方向ベクトル出力用)を追加。既存モードはデフォルト値のままなので後方互換。

```
python3 dwt.py --quick --stage all                       # 動作確認
python3 dwt.py --stage 1                                 # DNだけ学習
python3 dwt.py --stage 2                                 # WTNだけ学習(GTの方向を使用)
python3 dwt.py --stage 3                                 # 結合fine-tuning
python3 dwt.py --max-vram-gb 8 --max-gpu-util 0.8         # main.pyと同じGPU制限フラグ
```

### VRAM使用量について

Stage3(結合fine-tuning)はDN+WTN両方のactivationを同時に保持するため、単体学習より
メモリを多く使う。`--quick`(3フレーム)ですら`--batch-size`デフォルト(8)のままだと
VRAM 12GB環境でOOMした。そのため`--joint-batch-size`(未指定時は`--batch-size`の半分)
でStage3だけ別のバッチサイズを使えるようにしている。

## 動作確認 (2026-07-15, `--quick`)

3フレーム・各ステージ3エポックで、クラッシュせず3ステージとも完走することを確認:

```
Stage 1 (DN):    train_loss 3.22 → 1.56(あるいは2.04)
Stage 2 (WTN):   train_loss 2.78 → 2.01〜2.22
Stage 3 (Joint): train_loss 1.37 → 1.13(またはtrain_loss 1.63→1.41)
```

いずれも損失は単調に減少しており、学習として機能している。ただし3エポックでは
全く学習が足りておらず、推論結果は輪郭0個(空)だった。これは想定内(`--quick`は
「崩壊しないか」の確認用、[AFTERMATH_OBJECTIVE.md](AFTERMATH_OBJECTIVE.md)と同じ方針)。

## Fluo-N3DH-SIM+・Fluo-N3DH-CHOでの1epoch版 (2026-07-15〜16)

result.htmlに組み込み済み。Fluo-N3DH-SIM+は1epochでもGTとほぼ完全一致
(例: t010でGT=7、予測=7)。Fluo-N3DH-CHOはt005のGT4/11接触ペアがまだ分離できて
おらず(1epochでは不足)、[Cellpose](PROJECT_Cellpose.md)の方が先に成功した。

なお、CHOで「GT」と表示・比較していたものは実際にはST(Silver Truth)である点に
注意([PROJECT_Cellpose.md](PROJECT_Cellpose.md)の訂正セクション参照)。

## 本物のGT(19枚)での評価 (2026-07-16, `eval_real_gt.py`)

CTC公式のSEG measureで、公平に使えるtest分割内の本物GT2枚(t033, t062)のみで評価:

```
DWT(CHO, 1epochのみ):              SEG=0.576
Cellpose(ゼロショット、同じ2枚):     SEG=0.625
binary_DiffusionFlow(35epoch):     SEG=0.689
```

わずかにCellposeが上回り、さらにその後実装した`binary_DiffusionFlow`
(熱拡散ベースflow場、[PROJECT_DiffusionFlow.md](PROJECT_DiffusionFlow.md))が
両方を上回った。ただしDWTは1エポックのみの学習なので、本番学習(SIM+で開始したが
一時中断中)後に改めて再評価する必要がある。

## 今後

- [ ] 本番学習(フルフレーム・多エポック、SIM+は35epoch/patience5で開始したが
  一時中断中。再開が必要)を実行
- [ ] 本番学習後、`eval_real_gt.py --method dwt`で本物GT2枚に対して再評価し、
  Cellposeと比較する
- [ ] 既知の接触ペア(t005のGT id=4/11など)が実際に分離できているかを確認
  ([PROJECT_labeling3.md](PROJECT_labeling3.md)セクション10の「GT照合手段が無い」問題も
  合わせて解消する必要がある。ただしt005自体に本物のGTは無い点に注意)

# binary_DiffusionFlow: 熱拡散ベースflow場の自前学習 (2026-07-16)

## 背景

DWT・Cellposeの検討を経て、「STで学習しても本物のGTで評価すれば意味がある」
「距離の勾配(設計A)は前景+距離から事後計算できてしまい無駄だが、熱拡散ベース
(設計B)は前景だけからは事後計算できない(インスタンスの区切りが必要なため)」
という2点の整理がついたので、Fluo-N3DH-CHOのST(Silver Truth)を使って
(前景, 熱拡散ベースのflow場)を自前で学習することにした。詳細な設計議論は
本チャットログ、および[NEXTOBJECTIVE.md](NEXTOBJECTIVE.md)参照。

## 設計

- **出力**: (前景, dy, dx)の3チャンネルのみ(距離チャンネルは持たない、Cellpose本体の
  出力構成に近い)。`UNet(out_channels=3, activation="sigmoid_tanh2")`
  (前景=sigmoid、dy/dx=tanh。既存のUNetクラスがn=3でもそのまま使える)。
- **flow場の教師データ**: GT/STのインスタンスマスクの中で熱拡散シミュレーションを
  行い、**収束しきる前(反復回数=2×dt.max()²)で打ち切った**時点の温度分布の勾配。
  収束しきると全体が同じ温度になり意味のある勾配が消えるため、あえて途中で止める
  (詳細な理屈はPROJECT_Cellpose.mdの熱拡散の説明、および本チャットログのn_iterの
  議論を参照)。
- **モデルが学習するのは「生画像→(前景,dy,dx)」という写像のみ**。熱拡散
  シミュレーション自体はモデルの学習対象ではなく、教師データを作るための
  前処理(学習前に1回だけ実行)。推論時は熱拡散を一度も呼ばない。
- **損失**: `fg_loss(bce_dice) + dy_loss(MSE,前景のみ) + dx_loss(MSE,前景のみ)`。
- **推論**: 既存のdistance_flow_trainedと同じ軌跡積分+クラスタリング
  (`main._integrate_and_cluster`)をそのまま再利用。

## 実装

`diffusion_flow.py`に新規実装。

### 計算コストへの対応

熱拡散は反復回数がインスタンスの半径の2乗のオーダーで必要(このデータセットの
最大半径約44pxで2000〜4000反復程度)なので、毎エポック計算し直すと非現実的。
そのため、**train/val全フレーム分のflow場を学習前に1回だけ計算しディスクに
キャッシュする**(`precompute_cache()`、`cache/<dataset>/diffusion_flow/`)。
全インスタンスを1回のループで同時に処理する(同じインスタンスIDの隣接ピクセルと
だけ平均する)よう実装し、実測でCHO全体(410スライス)の事前計算は約18分だった。

```
python3 diffusion_flow.py --epochs 35 --patience 5 --max-vram-gb 8 --max-gpu-util 0.85
```

## 結果 (2026-07-16, Fluo-N3DH-CHO)

35エポック上限に対しEarly Stoppingせず完走(patience=5に達しなかった)。
train_loss 2.28→0.45、val_loss 1.62→0.51と単調に改善。

### 本物のGT(t033, t062)でのCTC SEGスコア比較(`eval_real_gt.py`)

```
DWT(CHO, 1epochのみ):              SEG=0.576
Cellpose(事前学習済み、ゼロショット): SEG=0.625
binary_DiffusionFlow(35epoch):     SEG=0.689  ← 最良
```

STで学習したにもかかわらず、本物のGTで評価してCellpose(ゼロショット)を上回った。

### 看板ケース(t005のGT id=4/11接触ペア)の分離

t005 z=0で目視確認したところ、GT id=4/11がタグ2つ(形・位置ともほぼ一致)に
正しく分離できていた。DWT(1epoch)は分離できていなかったが、Cellpose・
binary_DiffusionFlowはいずれも成功した(ただしt005自体に本物のGTは無く、
ST基準の確認である点に注意、[PROJECT_Cellpose.md](PROJECT_Cellpose.md)参照)。

## result.htmlへの反映

`diffusion_flow_track_and_visualize.py`(dwt_track_and_visualize.py・
cellpose_track_and_visualize.pyと同じ構成)で生成。①モデル列
「binary_DiffusionFlow」→「full_trained」→②「熱拡散flow場による軌跡積分」。

## 輪郭抽出の後処理案: 前景マスクのモルフォロジー・クロージング (2026-07-16, 未解決)

t051で、細胞内部の暗い模様のせいで前景確率マップに小さな"穴"(凹み)ができ、
その穴のせいで軌跡積分(`_integrate_and_cluster`)が1つの細胞を2つのタグに
分裂させてしまう例が見つかった。対策として、前景マスクに軌跡積分の前段で
モルフォロジー・クロージングをかけるオプション(`to_contour_input`の
`close_radius`引数、`--close-radius`でrun生成可能)を実装した。

**未解決(重要な方法論上の問題あり)**: t051(1例、しかもtestフレーム)だけを見て
radiusを調整したところ、radius=16でその箇所は局所的に直ったが、フレーム全体の
輪郭数は変わらなかった(=別の場所で新たに分裂/融合が起きた)。これは
①1例への過剰適合、②test set leakage(testフレームを見てチューニングしてはいけない)
という2つの問題を含む、恣意的なチューニングの実例だった。詳細な教訓は
[NEXTOBJECTIVE.md](NEXTOBJECTIVE.md)の「重要な教訓」セクション参照。

**正しくやり直すなら**: train/valフレーム(または本物のGTのうちtest外の17枚)で
複数例を使ってradiusを検証し、最後に1回だけtest(t033/t062)で確認する。

## 今後

- [ ] close_radiusをtrain/valフレームで正しく検証し直す(上記参照)
- [ ] Fluo-N3DH-SIM+でも同様に試す(このデータセットは本物のGTが密なので、
  より信頼できる比較ができる)
- [ ] 過剰予測気味な傾向(例: t061でGT=10、フィルタ後予測=15)の原因を調べる
- [ ] タグ付け(OT/UOT)の検討に戻る([NEXTOBJECTIVE.md](NEXTOBJECTIVE.md)参照)

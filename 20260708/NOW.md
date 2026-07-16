# 現状まとめ (2026-07-15)

## パイプライン全体

```
① 学習: 正解データの各ピクセルにチャンネルを割り当てて学習
   - (前景) → (前景,距離) → (前景,距離,dx,dy) と段階的に拡張して試行
② 推論: テストデータをモデルに通し、上記チャンネルの予測を得る
③ 輪郭構築: 予測から個々の細胞のインスタンスを作る
   - 前景の境界をそのまま輪郭にする
   - 距離を使ったwatershed
   - 距離を使った卓立度(prominence)フィルタ
```

現状の詳細な検討ログは[PROJECT_labeling3.md](PROJECT_labeling3.md)、
それ以前の経緯は[PROJECT_labeling.md](PROJECT_labeling.md)/
[PROJECT_labeling2.md](PROJECT_labeling2.md)参照。

## タグ付け（インスタンス識別）の状態

- Zスライス間のタグ一貫性づけ: **実装済み**（コスト=1-IoU＋ハンガリアン法、
  `track_contours_across_z`）
- ノイズタグの除去: **実装済み（あまり深入りせず）**（生涯平均面積が
  閾値150px未満のタグを除外。15テストフレーム中10フレームでGT完全一致）
- GTの個々のインスタンスとの直接照合: **未着手**（合計数が合っていても
  個別ペアが正しく分離できているかは別途確認が必要、
  [PROJECT_labeling3.md](PROJECT_labeling3.md)セクション10）
- 時間方向（フレーム間）の追跡: **未着手**（Z間のみ実装済み）

詳細は[PROJECT_tagging.md](PROJECT_tagging.md)参照。

## 「最もうまくいっている例」（Z間タグ+面積閾値）に残る未解決の問題

- スライスの途中でタグが分離した場合に対応できない
- 細胞がくっついていると対応できない（＝この文書全体の中心課題）

## 接触分離の各手法が行き詰まった理由

- **watershed**: ピーク検出(`peak_local_max`)は固定の空間しきい値
  (`min_distance=10px`)だけでピークの採否を決めており、細胞数を入力として
  要求するわけではない（＝「細胞数が先にないとピーク数が決まらない」という
  文字通りの循環論法ではない、誤りだったため訂正）。ただし本質的な問題は残る：
  そのしきい値に「本物の2細胞かノイズによる偶然の2山か」を判断する仕組みが
  一切なく、しきい値を動かしても過剰分割⇄見逃しのトレードオフにしかならない
  （[PROJECT_labeling3.md](PROJECT_labeling3.md)セクション1）
- **卓立度(prominence)フィルタ**: n=2の弱い確認に留まり、一般化できるか不明
  （同セクション5-6）
- **(前景,距離,dx,dy)の4ch設計**: dx,dyは前景・距離から導出可能で独立情報でなく、
  モデル通過後のdx,dyが距離の微分と一致する保証もないため、
  「一致するか」という議論自体が意味をなさない（同セクション3）
- **3次元化**: 検討はしたが、2次元でまず決着をつけたいという判断で保留中
  （同セクション2, 8）

## 次の方向性: OT/UOTによるタグ付け

先週のゼミでOT(最適輸送)/UOT(非平衡最適輸送)を使ったタグ付けの方向で
データセットを探していたが、細胞が接触している場合を考慮していなかったことに
気づいた。現在はまず接触した細胞を正しく分離する手法を探している段階
（本文書の他セクションの検討はすべてこの一環）。

## Deep Watershed Transform (DWT) の導入 (2026-07-15〜)

上記の「本物とノイズを区別する仕組みが無い」問題に対応する既存研究として
Deep Watershed Transform(Bai & Urtasun, CVPR 2017)を採用し実装した(`dwt.py`)。
Fluo-N3DH-SIM+では1epoch版でGTとほぼ完全一致するほど好結果。Fluo-N3DH-CHOの
1epoch版はt005のGT4/11接触ペアがまだ分離できておらず、本番学習(多エポック)待ち
（一時中断中、要再開）。詳細は[PROJECT_DWT.md](PROJECT_DWT.md)参照。

## Cellpose(事前学習済み、ゼロショット)の導入 (2026-07-15〜)

DWTと並行して、事前学習済みCellpose(cpsam_v2)を**このデータでの再学習なし**で
そのまま試した(`cellpose_track_and_visualize.py`)。Fluo-N3DH-CHOのt005で、
このプロジェクトが繰り返し失敗してきたGT4/11接触ペアの分離に**初めて成功**
（再学習ゼロにもかかわらず）。詳細・仕組みの説明は[PROJECT_Cellpose.md](PROJECT_Cellpose.md)参照。

## 重要な訂正: CHOの「GT」は実はST (2026-07-15)

Fluo-N3DH-CHOで普段「GT」として表示・比較していたマスクは、実際には**ST**
(Silver Truth、アルゴリズム生成の疑似正解、`main.py`の`annotation_subdir="ST"`)
だった。本物のGT/SEG(人手検証済み)は19個の(フレーム,Zスライス)にしか存在せず、
このプロジェクトが看板にしてきたt005のGT4/11には本物のGTが無い。
result.htmlは上部バナー・フレームサマリー等でGT/STを動的に正しく表示するよう
修正済み(build_viewer.py)。

## 本物のGT(t033,t062)でのSEGスコア比較 (2026-07-16 更新)

`eval_real_gt.py`(CTC公式のSEG measure)で、本物のGT19枚のうちtest分割内の
公平な2枚(t033,t062)を使って比較:

```
DWT(CHO, 1epochのみ):              SEG=0.576
Cellpose(事前学習済み、ゼロショット): SEG=0.625
binary_DiffusionFlow(35epoch):     SEG=0.689  ← 現時点で最良
```

`binary_DiffusionFlow`(前景+熱拡散ベースflow場、STで学習)がCellposeを上回った。
看板ケース(t005のGT4/11接触ペア)もCellpose・binary_DiffusionFlow両方が分離に
成功(DWT 1epochは未分離、ただしST基準の確認)。
詳細は[PROJECT_Cellpose.md](PROJECT_Cellpose.md)・[PROJECT_DWT.md](PROJECT_DWT.md)・
[PROJECT_DiffusionFlow.md](PROJECT_DiffusionFlow.md)参照。

## 重要な教訓: 恣意的なパラメータ調整とtest set leakage (2026-07-16)

輪郭抽出の後処理パラメータ(`close_radius`)を、testフレーム(t051)を1例だけ見て
調整してしまい、①1例への過剰適合、②test set leakage、という2つの問題が
判明した。「恣意的か合理的か」は(a)理論的な導出があるか、(b)train/valの
広いサンプルで検証されているか、で判断する。「機械学習ができた」とは、
testを一切見ずに選んだ設定がtestで良い性能を出すこと。詳細・今後の運用ルールは
[NEXTOBJECTIVE.md](NEXTOBJECTIVE.md)の「重要な教訓」セクション参照。

# DATA_CELL — データセット入手方法

このディレクトリの中身(`Fluo-N3DH-SIM+/`, `BF-C2DL-HSC/`)はサイズが大きいため
`.gitignore`で除外し、git管理には含めていません。このファイルだけを残して
おくので、必要な時は以下の手順で再取得してください。

## Fluo-N3DH-SIM+

- 出典: Cell Tracking Challenge (http://celltrackingchallenge.net/)
- ダウンロードURL: http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-SIM+.zip
- サイズ: 約3.6GB (展開後、01/ と 02/ の各シーケンス + ERR_SEG/GT)
- 取得スクリプト: `20260620/m_real.py`(`ensure_dataset()` 関数がwgetでダウンロード→展開まで自動化)
  - `DATASETS_DIR`は`DATA_CELL/`を指すので、スクリプトを再実行すればこのディレクトリに展開される
- 手動取得する場合:
  ```bash
  wget -c http://data.celltrackingchallenge.net/training-datasets/Fluo-N3DH-SIM+.zip -O DATA_CELL/Fluo-N3DH-SIM+.zip
  unzip DATA_CELL/Fluo-N3DH-SIM+.zip -d DATA_CELL/
  ```

## BF-C2DL-HSC

- 出典: Cell Tracking Challenge (http://celltrackingchallenge.net/)
- ダウンロードURL(推定 — Fluo-N3DH-SIM+と同じURL命名規則からの推測。実行前に
  http://celltrackingchallenge.net/ で正式なURLを確認してください):
  http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip
- サイズ: 約2.1GB (01/, 02/ の各シーケンス + ERR_SEG/GT/ST)
- 手動取得する場合:
  ```bash
  wget -c http://data.celltrackingchallenge.net/training-datasets/BF-C2DL-HSC.zip -O DATA_CELL/BF-C2DL-HSC.zip
  unzip DATA_CELL/BF-C2DL-HSC.zip -d DATA_CELL/
  ```

## 経緯

- 2026-06-20: `20260620/m_real.py`がFluo-N3DH-SIM+を`20260620/datasets/`にダウンロード
- 2026-06-25: `20260625/`の実験でFluo-N3DH-SIM+(同一内容)とBF-C2DL-HSCを`20260625/datasets/`に再取得
- 2026-07-02: 重複していたFluo-N3DH-SIM+を`diff -rq`で完全一致確認の上、`DATA_CELL/`に一本化。
  `20260620/datasets/`(重複分)は削除し、参照コードのパスを`DATA_CELL/`に更新

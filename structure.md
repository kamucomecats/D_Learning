# プロジェクト構造ガイド (`structure.md`)

このドキュメントは、本プロジェクトのディレクトリ構成と、それぞれのフォルダやファイルの役割を明確にすることを目的とします。

---

## 📁 プロジェクト全体構成
.
├── README.md
├── architecture.md
├── cnn
│   ├── train_nn.py
│   └── two_layer_net.py
├── common                         # 汎用的関数(ミニバッチの生成など)
│   ├── __pycache__
│   ├── batch.py                   # ミニバッチ生成
│   ├── common_functions.py        # Softmaxなどを格納
│   ├── grad.py                    # 数値微分のみ、多次元対応済み
│   ├── gradient_check.py          # 数値と逆伝播の勾配比較用
│   ├── mnist_load.py              # MNISTデータをone-hotでロード
│   └── optimizer.py               # SGD、Momentumを格納
├── dev-notes.md
├── mnist_local.npz
├── nn
│   ├── __pycache__
│   ├── train_nn.py                # 本体、opt, grad切り替え可
│   ├── Layers.py                  # 誤差逆伝播法のレイヤー定義
│   └── two_layer_net.py           # NNモデル
├── requirements.txt
├── structure.md
└── train_loss.png
---

## 📌 各ディレクトリの詳細

### `src/`

- **責務**: プロジェクトの本番用ロジックをすべて含む
- モジュール単位で分割されており、再利用可能性・保守性を意識して構成
- CLI 実行は `src/train.py` を入口とする

### `src/models/`

- ニューラルネットワーク構造の定義ファイル群
- クラス単位で定義し、任意のアーキテクチャに差し替え可能とする
- 活性化関数や初期化方法を差し替えられるように拡張性を持たせる方針

### `src/utils/`

- ミニバッチ生成、one-hot変換など、学習パイプラインに汎用的に使える処理
- 関数単位で提供され、依存関係を少なく保つことを意識

### `experiments/`

- 試験的な実装、アイディアの検証コード、テストスクリプト
- 実験用であり、最終的に `src/` に統合されるか削除される可能性あり
- ファイル冒頭に「目的」「使い方」「結果」など簡単なコメントを記載する


---

## 🧭 命名・ファイル構成の方針

- ファイル名は **snake_case**, クラス名は **CamelCase**
- 可能な限り **1ファイル＝1目的** を維持する
- モジュールの依存は **一方向に** 保つ（例：utils → models には依存しない）

---

## 🛠 今後の予定・拡張構想（任意）

- `tests/` ディレクトリを追加し、ユニットテストを自動化
- `configs/` に設定ファイル（YAML）を置いて再現性・柔軟性を確保
- `scripts/` を作成し、学習・評価・可視化などをCLI実行にまとめる

---

## 📎 関連ドキュメントリンク

- [architecture.md](architecture.md): アーキテクチャ設計・前提条件
- [dev-notes.md](dev-notes.md): 開発中のメモ・実験ログ

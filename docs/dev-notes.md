# 開発メモ・備考

## 開発ログ

- 以下、サンプル
- 2025-04-19：勾配法を偏微分にて導出。今後誤差逆伝播法への切り替えを検討。
- 2025-04-21：`batch.py` を追加。train loop からの分離に成功。
- 2025-04-24：学習中の損失・精度を `matplotlib` で出力するように変更。
- 以上、サンプル

## 注意点・トラブル対応

- 以下、サンプル
- `numpy.dot` において shape mismatch のエラーが発生しやすい → `print(x.shape)` デバッグ推奨
- `axis` の指定ミスによる平均計算の不整合に注意（特に softmax の場合）
- 以上、サンプル

## 今後のアイデア（ToDo）

<<<<<<< HEAD
- [x] softmax関数、cross_entropy_error関数、sigmoid関数の実装 (一旦 experiments/common_functions.pyへ)
- [ ] grad関数の実装(仮でベクトルに対しての勾配を導出する、のちに多次元に対応して実装)
- 以下、サンプル
- [ ] 重み初期化方式（He vs Xavier）の比較
- [ ] optimizer（SGD → Adam）切替実験
- [ ] 実験結果の保存形式（CSV / JSON / pickle）
- 以上、サンプル
=======
- [ ] softmax関数、cross_entropy_error関数、sigmoid関数の実装 (一旦 experiments/common_functions.pyへ)
- [ ] grad関数の実装
>>>>>>> 9e24b861f473c4b05deeb0451843837d83d4b42a

## 開発の大まかな項目

- [x] MNISTデータセットロード
- [x] バッチ処理
- [ ] 損失関数
- [ ] 勾配法
- [ ] 2層NNのクラス(ミニバッチまで)
- [ ] 誤差逆伝播法
- [ ] CNN
- [ ] その他学習上のテクニック
# tts_impl (仮)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3119/)


このリポジトリは制作中です。MITライセンスで利用できますが、予告なく内容が変更される可能性があります。

## インストール
```sh
pip install git+https://github.com/uthree/tts_impl.git
```
または、このリポジトリをクローンして
```sh
pip install -e .
```

## コード検査
`black src` でフォーマッタを実行。  
`pytest` でテストを実行。  
`mypy src` で型検査を実行。  


## TODO List
- pipでのインストール、何らかのプロジェクトへこのリポジトリを組み込んでの運用
- また、TTSのText Encoderに言語モデルの特徴量を入力する機構をつけたい。
- Onnxエクスポート: Python, PyTorchがない環境で推論できるようにしたい。(e.g. Rust+ort)
- "recipe": モデルアーキテクチャを指定して、データセットを準備すれば前処理から学習まですべてできる
ようにしたい。
- 音声データから自動書き起こし、話者分類、BGMやノイズ、無音区間の除去など、データセット制作を補助する機能

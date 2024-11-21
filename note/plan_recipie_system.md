# レシピシステム
## レシピとは
データセットの前処理手法、モデル構造、学習時のハイパーパラメーター等をひとまとまりにしたもの。 同じレシピとデータセットを用意すれば再現実験ができるようにしたい。
YAMLファイルまたはJSONファイルで記述できるようにしたい。  
Omegaconf, Hydraとの兼ね合いも考慮して、YAMLを採用する可能性が高い。

## モジュラー式モデル構造
ベースとなるモデルを用意し、そのサブモジュールを自由に差し替える。  
型チェックの恩恵を受けるため、[Protocol](https://typing.readthedocs.io/en/latest/spec/protocol.html)の詳細な実装を用意してある。  

やっていることとしてはJavaにおける「インターフェース」、Rustにおける「トレイト」に近い。
内部の挙動に関係なく共通の振る舞いを定義することで、同じ振る舞いの別のものに差し替えることができる。

~~正直、 この設計思想はPythonのような動的言語でやるようなものではないのはわかっているが、こうでもしないと少し構造を変更するたびに変更すべき箇所が多すぎてまともに実験ができないのだ。~~

### モデル組み換え例
例えば、VITS2をベースにデコーダー部分をBigVGANに変更、length reguratorをgaussian upsamplingにする、といった処理をする。  
ライブラリを使う側としては、差し替えるモデルを選択or実装するだけで簡単に複合モデルを設計できる

```py
#警告: このコードは構想・計画段階のメモです。実際に動作するわけではありません。

from tts_impl.net.tts.vits import VitsGenerator

custom_vits = VitsGenerator(
    vocoder="bigvgan", # default = "hifigan"
    length_regurator="gaussian", # default = "duplicate"
)
```

## 前処理

## 実験と評価
# TODO list
- HiFi-GAN系統のモデルを統合・抽象化する。
    - Generator, Discriminatorの処理を分けるなど
    - 対象となるモデル
        - HiFi-GAN
        - NSF-HiFi-GAN
        - FC-HiFi-GAN
        - iSTFTNet
        - MS-HiFi-GAN
        - MB-HiFi-GAN
        - EVA-GAN
        - SAN

- VITS, JITSに共通するmonotonic alignを統一する
- ヘルパー関数の一般化・整備
    - sequence_mask, generate_path etc...

- VITSのカスタムをサポートする?
    - f0
    - 言語モデル特徴量
    - length_regurator
    - decoder
    - VITS2との統合管理

- g2p
    - pyopenjtalk
    - phonemizer
- ドキュメントをいい感じに
    - docstringを解析して自動生成が望ましいかも。
- Recipe
    - ハイパーパラメータやデータの前処理や学習の流れなどをまとめたクラス。
- transformersとの連携
- CLI

- モデルの分類の見直し?
    - end-to-endなTTSとメルスペクトルを経由するTTSを同一のカテゴリにして良いのか？
    - そもそも分類が必要か？
        - VITSはTTS/VCが可能なのでTTSモデルなのかVCモデルなのか明確に決められない?

- ハイパーパラメータをdataclass + omegaconfで管理する
    - Optunaとの連携

- データセットのメタデータをいい感じにする。
    - サンプリングレート, 話者一覧など。

- Logger
    - tqdmを直接呼ぶ形式ではなくもっと抽象化する+richべーすにする。
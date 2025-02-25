# TODO list
- VITS, JITSに共通するmonotonic alignを統一する

- ヘルパー関数の一般化・整備
    - sequence_mask, generate_path etc...

- g2p
    - phonemizer
    - BERT-CTC-G2P

- ドキュメントをいい感じに
    - docstringを解析して自動生成が望ましいかも。

- Recipe
    - ハイパーパラメータやデータの前処理や学習の流れなどをまとめたクラス。
    - 複数のLightningModuleに対応できるように再実装する (refactor/rewrite)

- transformersとの連携

- モデルの分類の見直し?
    - end-to-endなTTSとメルスペクトルを経由するTTSを同一のカテゴリにして良いのか？
    - そもそも分類が必要か？
        - VITSはTTS/VCが可能なのでTTSモデルなのかVCモデルなのか明確に決められない?

- ハイパーパラメータをdataclass + omegaconfで管理する
    - Optunaとの連携
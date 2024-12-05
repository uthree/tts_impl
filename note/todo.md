# TODO list
- VITS, JITSに共通するmonotonic alignを統一する

- ヘルパー関数の一般化・整備
    - sequence_mask, generate_path etc...

- VITSのカスタムをサポートする?
    - 言語モデル特徴量
    - length_regurator
    - decoder
    - VITS2との統合管理...?
    - 戻り値の変数の数が多すぎて扱いにくいので、辞書にするとかしたほうがいい？

- g2p
    - pyopenjtalk
    - phonemizer
    - BERT-CTC-G2P

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

- Configuration
    - コンフィグデータクラスの自動導出と初期化、辞書への自動変換などを実装したが、複雑化してしまった。マクロ的なことをPythonでやると可読性が著しく低下するうえに型チェックの恩恵が得られない！！
# tts_impl (仮)
**わーくいんぷろぐれす**

implementation of TTS models in PyTorch  
TTSモデルのPyTorch実装集　なるべく依存関係を減らす
pytorch lightningを使って学習できるようにする。これにより簡単に複数GPUを利用でき、TensorBoardでの進捗監視ができる。  
モデル構造がほぼ同じなので、VITS派生の(S)VCもついでに実装。


## テスト
`pytest` でテストを実行できる。

## 各ディレクトリの詳細
- `src/tts_impl/net/` : ネットワークアーキテクチャの定義
    - `vocoder/` 音響特徴量から音声波形を生成するモデル。
        - `hifigan/`: HiFi-GANの実装。

## TODO List
- 学習処理
- pipでのインストール、何らかのプロジェクトへこのリポジトリを組み込んでの運用
- モジュラー設計
    - end-to-endなTTSなど、ボコーダー部分を任意に差し替えることができる機構を備えたい
    - 共通のインターフェイスを多くのボコーダーに実装すれば可能そう。
    - ボコーダーに関してはDiscriminatorを差し替えたりなど
- また、TTSのText Encoderに言語モデルの特徴量を入力する機構をつけたい。
- Onnxエクスポート: Python, PyTorchがない環境で推論できるようにしたい。(e.g. Rust+ort)
- "recipe": モデルアーキテクチャを指定して、データセットを準備すれば前処理から学習まですべてできる
ようにしたい。
- 公式実装の学習済みモデルからインポート


### モデル一覧
✅ : 実装済み
🚧 : 実装着手中 
❓ : 計画・構想中

Vocoder:
- HiFi-GAN ✅
- HiFi-GAN Variants(NSF, Harmonic, SiFi-GAN, EVA-GAN) ❓
- more discriminators(CQT, MRSD) ❓
- BigVGAN ❓
- DDSP❓

TTS: text to speech
- via mel spectrogram
    - FastSpeech2 🚧
- end-to-end
    - VITS ❓
    - JETS 🚧

linguistic frontend
- g2p:
    - pyopenjtalk-plus ❓
    - phonemizers ❓
- alignment:
    - on-tye-fly alignment(monotonic-alignment-search, forward-sum, etc.) 🚧
    - Montreal Forced Aligner
- feature adapter
    - BERT / RoBERTa ❓
    - predict accent classic (e.g. dictionary) method ❓
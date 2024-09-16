# tts_impl (仮)
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
- pipでのインストール、何らかのプロジェクトへこのリポジトリを組み込んでの運用
- モジュラー設計
    - end-to-endなTTSなど、ボコーダー部分を任意に差し替えることができる機構を備えたい
    - 共通のインターフェイスを多くのボコーダーに実装すれば可能そう。
    - ボコーダーに関してはDiscriminatorを差し替えたりなど
- また、TTSのText Encoderに言語モデルの特徴量を入力する機構をつけたい。
- Onnxエクスポート: Python, PyTorchがない環境で推論できるようにしたい。
- モデルアーキテクチャを指定して、データセットを準備すれば前処理から学習まですべてできる
ようにしたい。


### モデル一覧
"✅" がついているものは実装済み。

Vocoder / Codec:
- HiFi-GAN ✅
- NSF-HiFi-GAN
- SiFi-GAN
- BigVGAN
- UnivNet
- DDSP(combtooth, sinusoidal)
- SAN Discriminators
- MRSD, CQT Discriminator
- WaveGrad

TTS: text to speech
- VITS
- JETS
- FastSpeech2
- Glow-TTS
- Tacotoron
- Diffusion, ODE系

SVS: Singing Voice Synthesis
- VISinger / VISinger 2

VC: Voice Conversion
- So-VITS-SVC
- RVC
- DDSP-SVC
- Cyclone

PE: Pitch Estimation
- CREPE
- RMVPE
- FCPE

grapheme to phoneme frontend: 
- pyopenjtalk-plus
- phonemizers

language model frontend:
- BERT / RoBERTa, w/Tokenizer

Evaluation:
- SpeechMOS
- UTMOS
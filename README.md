# tts_impl
implementation of TTS models in PyTorch  
TTSモデルのPyTorch実装集　なるべく依存関係を減らす

## Note
- `module/net/` : ネットワークアーキテクチャの定義
    - `acoustic_model/`: メルスペクトログラムなどの音響特徴量を生成するモデル。
        - `fastspeech2.py`
    - `e2e_tts/` テキストから音声波形の生成までを行うモデル。
    - `vocoder/` 音響特徴量から音声波形を生成するモデル。
        - `hifigan/`: HiFi-GANの実装。
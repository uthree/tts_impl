# tts_impl.net以下のモジュール一覧
PyTorchの`torch.nn.module`または Torch Lightningの`lightning.LightningModule`として利用できるモジュール。  
ベースクラスとして[Protocol](https://typing.readthedocs.io/en/latest/spec/protocol.html)を定義している。

- aligner: アライメントモジュール。

- vocoder: 音響特徴量から音声波形を推定
    - [HiFi-GAN](https://arxiv.org/abs/2010.05646): `tts_impl.net.vocoder.hifigan`
        - LightningModule: `tts_impl.net.vocoder.hifigan.Hifigan`
        - Generator: `tts_impl.net.vocoder.hifigan.HifiganGenerator`
        - Discriminator: `tts_impl.net.vocoder.hifigan.HifiganDiscriminator`
    - HnNSF-HiFi-GAN: `tts_impl.net.vocoder.nsf_hifigan`
        - Generator `tts_impl.net.vocoder.nsf_hifigan.NsfhifiganGenerator`
    - Discriminator: `tts_impl.net.vocoder.discriminator`
    - DDSP
        - [Ultra-Lighweight DDSP](https://arxiv.org/abs/2401.10460)

- tts: Text-To-Speech, 音声読み上げ
    - [VITS](https://arxiv.org/abs/2106.06103): `tts_impl.net.tts.vits`
        - LightningModule: `tts_impl.net.tts.vits.Vits`
        - Generator: `tts_impl.net.tts.vits.VitsGenerator`
    - [JETS](https://arxiv.org/abs/2203.16852): `tts_impl.net.tts.jets`

- pe: ピッチ推定

- vc: 声質変換

- codec: コーデック

- g2p: grapheme-to-phoneme, テキストから音素列を推定するモデル。
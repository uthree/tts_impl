# tts_impl.net以下のモジュール一覧
PyTorchの`torch.nn.module`または Torch Lightningの`lightning.LightningModule`として利用できるモジュール。  
ベースクラスとして[Protocol](https://typing.readthedocs.io/en/latest/spec/protocol.html)を定義している。

- vocoder: 音響特徴量から音声波形を推定
    - ベースクラス群: `tts_impl.net.vocoder.base`
    - [HiFi-GAN](https://arxiv.org/abs/2010.05646): `tts_impl.net.vocoder.hifigan`
        - LightningModule: `tts_impl.net.vocoder.hifigan.Hifigan`
        - Generaotr: `tts_impl.net.vocoder.hifigan.HifiganGenerator`
        - Discriminator: `tts_impl.net.vocoder.hifigan.HifiganDiscriminator`
    - HnNSF-HiFi-GAN: `tts_impl.net.vocoder.nsf_hifigan`
        - Generator `tts_impl.net.vocoder.nsf_hifigan.NsfhifiganGenerator`
    - Discriminator: `tts_impl.net.vocoder.discriminator`

- TTS: Text-To-Speech, 音声読み上げ
    - ベースクラス群: `tts_impl.net.tts.base`
    - [VITS](https://arxiv.org/abs/2106.06103): `tts_impl.net.tts.vits`
        - LightningModule: `tts_impl.net.tts.vits.Vits`
        - Generator: `tts_impl.net.tts.vits.VitsGenerator`
    - [JETS](https://arxiv.org/abs/2203.16852): `tts_impl.net.tts.vits`
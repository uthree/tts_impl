# tts_impl
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3119/)

pytorch implementation of speech synthesis methods

## Models
implemented: ✅  
work in progress: 🚧  

- Vocoders
    - HiFi-GAN ✅
    - NSF-HiFi-GAN ✅
    - ISTFTNet ✅
    - WaveNeXt 🚧
    - Vocos 🚧
    - DDSP (Subtractive / Additive)
- TTS: Text to speech
    - VITS ✅
    - VITS2 🚧
    - JETS 🚧
    - FastSpeech
    - Diffusion TTS Models
    - Flow Matching TTS Modelss
- Aligner:
    - ForcedAlign ✅

## Installation
```sh
pip install git+https://github.com/uthree/tts_impl.git
```
or clone this repository,
```sh
pip install -e .
```

## type check / lint / test
- `pysen run format` : automatic formatting
- `pysen run lint` : linter
- `pytest` : test
- `codespell` : detect typos

## References
- [VITS](https://github.com/jaywalnut310/vits) ([Paper](https://github.com/espnet/espnet))
- [VITS2](https://github.com/p0p4k/vits2_pytorch/) ([Paper](https://arxiv.org/abs/2307.16430))
- [HiFi-GAN](https://github.com/jik876/hifi-gan) ([Paper](https://arxiv.org/abs/2010.05646))
- [Super Monotonic Alignment Search](https://github.com/supertone-inc/super-monotonic-align) ([Paper](https://arxiv.org/abs/2409.07704))
- [JETS (ESPNet)](https://github.com/espnet/espnet)

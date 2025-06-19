import torch
import torch.nn as nn
import torch.nn.functional as F
from tts_impl.net.base.stateful import StatefulModule
from tts_impl.net.common.grux import Grux
from typing import Optional, Tuple
from tts_impl.functional.ddsp import sinusoidal_harmonics
from tts_impl.net.base import GanVocoderGenerator


class NsfgruxFilterModule(StatefulModule):
    def __init__(
        self,
        in_channels: int,
        n_fft: int = 1024,
        frame_size: int = 256,
        d_model: int = 256,
        num_layers: int = 6,
        kernel_size: int = 4,
        d_ffn: Optional[int] = None,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.frame_size = frame_size
        self.gin_channels = gin_channels
        self.pre = nn.Linear(in_channels + n_fft + 2, d_model)
        self.post = nn.Linear(d_model, n_fft + 2)
        self.grux = Grux(
            d_model=d_model,
            num_layers=num_layers,
            kernel_size=kernel_size,
            d_ffn=d_ffn,
            d_condition=gin_channels,
        )

    def _parallel_forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pre(x)
        x, h = self.grux(x, h, c=g)
        x = self.post(x)
        return x, h

    def _initial_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.grux._initial_state(x)


class NsfgruxSourceModule(nn.Module):
    def __init__(
        self,
        sample_rate: int = 24000,
        num_harmonics: int = 1,
        frame_size: int = 256,
        noise_scale: float = 0.01,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.num_harmonics = num_harmonics
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.noise_scale = noise_scale
        self.gin_channels = gin_channels
        self.register_buffer("hann_window", torch.hann_window(n_fft))
        if gin_channels > 0:
            self.amps = nn.Parameter(torch.zeros(1, num_harmonics, 1))
        else:
            self.to_amps = nn.Conv1d(gin_channels, num_harmonics, 1)
            with torch.no_grad():
                self.to_amps.weight.zero_()

    def get_amps(self, g: Optional[torch.Tensor]) -> torch.Tensor:
        if self.gin_channels > 0:
            amps = F.normalize(torch.exp(self.amps), dim=1, p=2.0)
        else:
            assert g is not None
            amps = F.normalize(torch.exp(self.amps), dim=1, p=2.0)
        return amps

    def forward(
        self,
        f0: torch.Tensor,
        uv: Optional[torch.Tensor],
        g: Optional[torch.Tensor] = None,
    ):
        amps = self.get_amps(g)
        harmonics = sinusoidal_harmonics(
            f0, self.num_harmonics, self.sample_rate, hop_length=self.frame_size
        )
        if uv is None:
            uv = (f0 > 20.0).float()
        uv: torch.Tensor = F.interpolate(
            uv[:, None, :], scale_factor=self.frame_size, mode="linear"
        )
        harmonics_part = torch.sum(harmonics * amps, dim=1)
        noise_part = torch.randn_like(harmonics_part) * 0.333333
        harmonics_part += noise_part * self.noise_scale
        signal = noise_part * (1 - uv) + harmonics_part * uv
        return signal


class NsfgruxGenerator(nn.Module, GanVocoderGenerator):
    def __init__(
        self,
        source_module: NsfgruxSourceModule.Config = NsfgruxSourceModule.Config(),
        filter_module: NsfgruxFilterModule.Config = NsfgruxFilterModule.Config(),
        n_fft: int = 1024,
        frame_size: int = 256,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.n_fft = n_fft
        self.source_module = NsfgruxSourceModule(**source_module)
        self.filter_module = NsfgruxFilterModule(**filter_module)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        uv: Optional[torch.Tensor] = None,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # oscillate source signal
        src = self.source_module.forward(f0=f0, uv=uv, g=g)
        src_feat = self.analyze(src)  # STFT

        # concatenate input features and STFT features
        x = torch.cat([x, src_feat], dim=1)

        # filter forward pass
        x, _h_out = self.filter_module.forward(x=x)

        # synthesize waveform
        x = self.synthesize(x)

        return x

    def analyze(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.shape[1] == 1
        waveform = waveform.squeeze(1)
        h = self.frame_size // 2
        waveform = waveform[:, h:-h]
        wf_stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.frame_size,
            window=self.window,
            return_complex=True,
        )
        feats = torch.cat([wf_stft.real, wf_stft.imag], dim=1)
        return feats

    def synthesize(self, feats: torch.Tensor) -> torch.Tensor:
        real, imag = feats.chunk(2, dim=1)
        wf_stft = torch.complex(real, imag)
        waveform = torch.istft(
            wf_stft, n_fft=self.n_fft, hop_length=self.frame_size, window=self.window
        )
        h = self.frame_size // 2
        waveform = F.pad(waveform.unsqueeze(1), (h, h))
        return waveform

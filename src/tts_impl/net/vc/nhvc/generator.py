import math

import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from tts_impl.net.base.stateful import StatefulModule, StatefulModuleSequential
from tts_impl.net.common.mingru import MinGRU
from tts_impl.net.vocoder.ddsp import HomomorphicVocoder
from tts_impl.utils.config import derive_config


@derive_config
class NhvcEncoder(StatefulModule):
    """
    NHVC Encoder, this module encodes phoneme without speaker-specific information, and estimate pitch, noise-gate.
    """

    def __init__(
        self,
        in_channels: int = 80,
        d_model: int = 128,
        n_layers: int = 4,
        n_fft: int = 1024,
        d_phonemes: int = 64,
        n_phonemes: int = 500,
        n_f0_classes: int = 128,
        fmin: float = 20.0,
        fmax: float = 8000.0,
    ):
        super().__init__()
        self.d_phonemes = d_phonemes
        self.n_f0_classes = n_f0_classes
        self.fft_bin = n_fft // 2 + 1
        self.pre = nn.Linear(in_channels, d_model)
        self.fmin = fmin
        self.fmax = fmax
        self.stack = StatefulModuleSequential(
            [MinGRU(d_model) for _ in range(n_layers)]
        )
        self.post = nn.Linear(d_model, d_phonemes + n_f0_classes + self.fft_bin)
        self.to_phoneme_prob = nn.Linear(d_phonemes, n_phonemes, bias=False)

    def _initial_state(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.stack._initial_state(x, *args, **kwargs)

    def _parallel_forward(
        self, x: Tensor, h: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        x = self.pre(x)
        x, h_last = self.stack._parallel_forward(x, h, *args, **kwargs)
        x = self.post(x)
        return x, h_last

    def _sequential_forward(
        self, x: Tensor, h: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        x = self.pre(x)
        x, h_last = self.stack._sequential_forward(x, h, *args, **kwargs)
        x = self.post(x)
        return x, h_last

    def freq2idx(self, freq: Tensor) -> Tensor:
        log_fmin = math.log(self.fmin)
        log_fmax = math.log(self.fmax)
        log_delta_f = log_fmax - log_fmin

        log_freq = torch.log(torch.clamp(freq.float(), min=self.fmin, max=self.fmax))
        # Map to [0, n_f0_classes-2] range (since we have n_f0_classes-1 F0 classes)
        idx = torch.round(
            (log_freq - log_fmin) / log_delta_f * (self.n_f0_classes - 2)
        ).long()
        # Clamp to valid range
        idx = torch.clamp(idx, min=0, max=self.n_f0_classes - 2)
        return idx

    def idx2freq(self, idx: Tensor) -> Tensor:
        log_fmin = math.log(self.fmin)
        log_fmax = math.log(self.fmax)
        log_delta_f = log_fmax - log_fmin

        # Map from [0, n_f0_classes-2] to [fmin, fmax]
        log_freq = log_fmin + (idx.float() / (self.n_f0_classes - 2)) * log_delta_f
        freq = torch.exp(log_freq)
        return freq

    def decode_f0(self, probs: Tensor, k: int = 2) -> Tensor:
        """
        Args:
            probs: shape=[batch_size, n_f0_classes, n_frames]

        Returns:
            f0: shape=[batch_size, n_frames] in Hz
        """
        uv, f0_probs = torch.split(probs, [1, self.n_f0_classes - 1], dim=1)
        topk_result = torch.topk(f0_probs, k=k, dim=1)
        uv = (uv > 0.0).float().squeeze(1)

        # Convert indices to frequencies
        indices = topk_result.indices  # [B, k, T]
        freqs = self.idx2freq(indices)  # [B, k, T] in Hz

        # Weighted average using softmax probabilities
        probs = torch.softmax(topk_result.values, dim=1)  # [B, k, T]
        f0 = (freqs * probs).sum(dim=1) * uv  # [B, T]
        return f0

    def f0_loss(self, probs: Tensor, f0) -> Tensor:
        """
        Args:
            probs: shape=[batch_size, n_f0_classes, n_frames]
            f0: shape=[batch_size, n_frames]

        Returns:
            loss: shape=[]
        """
        uv = (f0 > self.fmin).float()
        uv_hat_logits, f0_logits = torch.split(
            probs.float(), [1, self.n_f0_classes - 1], dim=1
        )
        uv_hat = torch.sigmoid(uv_hat_logits.squeeze(1))
        loss_uv = (uv - uv_hat).abs().mean()
        f0_label = self.freq2idx(f0)
        loss_f0 = F.cross_entropy(f0_logits, f0_label)
        return loss_f0 + loss_uv


@derive_config
class NhvcDecoder(StatefulModule):
    """
    NHVC Decoder, this module reconstructs audio from phoneme embeddings and estimates HomomorphicVocoder parameters.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_fft: int = 1024,
        d_phonemes: int = 64,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.pre = nn.Linear(d_phonemes, d_model)
        self.gin_channels = gin_channels

        # Only use conditioning if gin_channels > 0
        if gin_channels > 0:
            self.stack = StatefulModuleSequential(
                [MinGRU(d_model, d_cond=gin_channels) for _ in range(n_layers)]
            )
        else:
            self.stack = StatefulModuleSequential(
                [MinGRU(d_model) for _ in range(n_layers)]
            )

        self.fft_bin = n_fft // 2 + 1
        self.post = nn.Linear(d_model, self.fft_bin * 2)
        self.d_phonemes = d_phonemes

    def _initial_state(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.stack._initial_state(x, *args, **kwargs)

    def _parallel_forward(
        self, x: Tensor, h: Tensor, g: Tensor | None = None, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        x = self.pre(x)
        if self.gin_channels > 0:
            x, h_last = self.stack._parallel_forward(x, h, cond=g, *args, **kwargs)
        else:
            x, h_last = self.stack._parallel_forward(x, h, *args, **kwargs)
        x = self.post(x)
        return x, h_last

    def _sequential_forward(
        self, x: Tensor, h: Tensor, g: Tensor | None = None, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        x = self.pre(x)
        if self.gin_channels > 0:
            x, h_last = self.stack._sequential_forward(x, h, cond=g, *args, **kwargs)
        else:
            x, h_last = self.stack._sequential_forward(x, h, *args, **kwargs)
        x = self.post(x)
        return x, h_last


@derive_config
class NhvcGenerator(StatefulModule):
    """
    NHVC Generator combining Encoder and Decoder with HomomorphicVocoder for voice conversion.
    """

    def __init__(
        self,
        encoder: NhvcEncoder.Config = NhvcEncoder.Config(),
        decoder: NhvcDecoder.Config = NhvcDecoder.Config(),
        vocoder: HomomorphicVocoder.Config = HomomorphicVocoder.Config(),
        n_speakers: int = 0,
        gin_channels: int = 0,
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.encoder = NhvcEncoder(**encoder)

        # Update decoder config with gin_channels if not already set
        if isinstance(decoder, dict):
            decoder_config = decoder.copy()
            if "gin_channels" not in decoder_config:
                decoder_config["gin_channels"] = gin_channels
            self.decoder = NhvcDecoder(**decoder_config)
        else:
            if not hasattr(decoder, "gin_channels") or decoder.gin_channels is None:
                decoder.gin_channels = gin_channels
            self.decoder = NhvcDecoder(**decoder)

        # Update vocoder config with sample_rate if not already set
        if isinstance(vocoder, dict):
            vocoder_config = vocoder.copy()
            if "sample_rate" not in vocoder_config:
                vocoder_config["sample_rate"] = sample_rate
            self.vocoder = HomomorphicVocoder(**vocoder_config)
        else:
            if not hasattr(vocoder, "sample_rate") or vocoder.sample_rate is None:
                vocoder.sample_rate = sample_rate
            self.vocoder = HomomorphicVocoder(**vocoder)
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.sample_rate = sample_rate

        if n_speakers > 0 and gin_channels > 0:
            self.speaker_embedding = nn.Embedding(n_speakers, gin_channels)

    def _initial_state(self, x: Tensor, *args, **kwargs) -> tuple[Tensor, Tensor]:
        h_enc = self.encoder._initial_state(x, *args, **kwargs)
        h_dec = self.decoder._initial_state(x, *args, **kwargs)
        return (h_enc, h_dec)

    def _parallel_forward(
        self,
        x: Tensor,
        h: tuple[Tensor, Tensor],
        sid: Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Args:
            x: Input acoustic features [batch_size, in_channels, n_frames]
            h: Hidden states (h_enc, h_dec)
            sid: Speaker IDs [batch_size] (optional)

        Returns:
            waveform: Synthesized audio [batch_size, 1, n_frames * hop_length]
            h_last: Updated hidden states (h_enc_last, h_dec_last)
        """
        h_enc, h_dec = h

        # Encode: get phoneme embeddings, f0 probabilities, and noise gate
        x = x.transpose(1, 2)  # [B, T, C]
        encoded, h_enc_last = self.encoder._parallel_forward(x, h_enc, *args, **kwargs)

        # Split encoder output
        phoneme_emb, f0_probs, noise_gate_logits = torch.split(
            encoded,
            [self.encoder.d_phonemes, self.encoder.n_f0_classes, self.encoder.fft_bin],
            dim=-1,
        )

        # Decode f0 from probabilities
        f0 = self.encoder.decode_f0(f0_probs.transpose(1, 2))  # [B, T]

        # Get speaker embedding if available
        g = None
        if sid is not None and self.n_speakers > 0:
            g = self.speaker_embedding(sid).unsqueeze(1)  # [B, 1, gin_channels]

        # Decode: get vocoder parameters
        decoded, h_dec_last = self.decoder._parallel_forward(
            phoneme_emb, h_dec, g=g, *args, **kwargs
        )

        # Split decoder output into env_per and env_noi
        decoded = decoded.transpose(1, 2)  # [B, fft_bin*2, T]
        env_per, env_noi = torch.split(decoded, self.encoder.fft_bin, dim=1)
        env_per = torch.exp(env_per)  # Convert to magnitude
        env_noi = torch.exp(env_noi)

        # Synthesize waveform using HomomorphicVocoder
        waveform = self.vocoder(f0=f0, env_per=env_per, env_noi=env_noi)
        waveform = waveform.unsqueeze(1)  # [B, 1, T*hop_length]

        return waveform, (h_enc_last, h_dec_last)

    def _sequential_forward(
        self,
        x: Tensor,
        h: tuple[Tensor, Tensor],
        sid: Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Sequential forward for streaming inference"""
        h_enc, h_dec = h

        # Encode
        x = x.transpose(1, 2)  # [B, T, C]
        encoded, h_enc_last = self.encoder._sequential_forward(
            x, h_enc, *args, **kwargs
        )

        # Split encoder output
        phoneme_emb, f0_probs, noise_gate_logits = torch.split(
            encoded,
            [self.encoder.d_phonemes, self.encoder.n_f0_classes, self.encoder.fft_bin],
            dim=-1,
        )

        # Decode f0
        f0 = self.encoder.decode_f0(f0_probs.transpose(1, 2))

        # Get speaker embedding
        g = None
        if sid is not None and self.n_speakers > 0:
            g = self.speaker_embedding(sid).unsqueeze(1)

        # Decode
        decoded, h_dec_last = self.decoder._sequential_forward(
            phoneme_emb, h_dec, g=g, *args, **kwargs
        )

        # Split and synthesize
        decoded = decoded.transpose(1, 2)
        env_per, env_noi = torch.split(decoded, self.encoder.fft_bin, dim=1)
        env_per = torch.exp(env_per)
        env_noi = torch.exp(env_noi)

        waveform = self.vocoder(f0=f0, env_per=env_per, env_noi=env_noi)
        waveform = waveform.unsqueeze(1)

        return waveform, (h_enc_last, h_dec_last)

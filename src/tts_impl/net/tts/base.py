from typing import Optional, Protocol

import torch


class Upsampler(Protocol):
    """
    protocol for upsampling encoded text by duration

    inputs:
        hs: [batch_size, channels, max_length]
        ds: [batch_size, max_length]
        h_masks: [batch_size, 1, max_length]
        d_masks: [batch_size, 1, max_length]
    outputs:
        features: [batch_size, channels, length_upsampled]
    """

    def forward(
        self,
        hs: torch.Tensor,
        ds: torch.Tensor,
        h_masks: Optional[torch.Tensor] = None,
        d_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class TextEncoder(Protocol):
    """
    A protocol for text encoders.
    """

    def __init__(self, vocab_size: int, out_channels: int, condition_dim: int = 0):
        """
        vocab_size: Vocabulary size of encodable tokens
        out_channel: Embedding dimensions per token
        condition_dim: Dimensionality of the condition (e.g. speaker embedding). Set to 0 to ignore the condition.
        """
        pass

    def forward(
        self,
        tokens: torch.Tensor,
        lenghts: Optional[torch.Tensor],
        condition: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        inputs:
            tokens: [batch_size, max_length], dtype=int
            lengths: [batch_size], dtype=float
            condition: [batch_size, 1, condition_dim] dtype=float
        returns:
            z: encoded features [batch_size, out_hannels, max_length], dtype=float
            mask: [batch_size, 1, max_length], 1=not masked, 0=masked
        """
        raise NotImplementedError


class AcousticFeatureEncoder(Protocol):
    """
    A protocol for acoustic feature encoders.
    """

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int = 0):
        """
        in_channels: The number of feature channels, such as fft_bin for spectrograms.
        out_channel: Embedding dimensions per token
        condition_dim: Dimensionality of the condition (e.g. speaker embedding). Set to 0 to ignore the condition.
        """
        pass

    def forward(
        self,
        features: torch.Tensor,
        lenghts: Optional[torch.Tensor],
        condition: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        inputs:
            features: [batch_size, channels, max_length], dtype=float
            lengths: [batch_size], dtype=float
            condition: [batch_size, 1, condition_dim] dtype=float
        returns:
            z: encoded features [batch_size, channels, max_length], dtype=float
            mask: [batch_size, 1, max_length], 1=not masked, 0=masked
        """
        raise NotImplementedError


class VariationalTextEncoder(Protocol):
    """
    A protocol for variational text encoders.
    This protocol is intended to be used primarily as a TextEncoder for VITS-based models.
    """

    def __init__(self, vocab_size: int, out_channels: int, condition_dim: int = 0):
        """
        vocab_size: Vocabulary size of encodable tokens
        out_channel: Embedding dimensions per token
        condition_dim: Dimensionality of the condition (e.g. speaker embedding). Set to 0 to ignore the condition.
        """
        pass

    def forward(
        self,
        features: torch.Tensor,
        lenghts: Optional[torch.Tensor],
        condition: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        inputs:
            phonemes: [batch_size, max_length], dtype=int
            lengths: [batch_size], dtype=float
            condition: [batch_size, 1, condition_dim] dtype=float
        returns:
            z: variational encoded features [batch_size, channels, max_length], dtype=float
            mean: [batch_size, channels, max_length], dtype=float
            log_s: log-scaled standard deviation, [batch_size, channels, max_length], dtype=float
            mask: [batch_size, 1, max_length], 1=not masked, 0=masked
        """
        raise NotImplementedError


class VariationalAcousticFeatureEncoder(
    Protocol,
):
    """
    A protocol for variational acoudic feature encoders.
    This protocol is intended to be used primarily as a PosteriorEncoder for VITS-based models.
    """

    def __init__(self, in_channels: int, out_channels: int, condition_dim: int = 0):
        """
        in_channels: The number of feature channels, such as fft_bin for spectrograms.
        out_channel: Embedding dimensions per token
        condition_dim: Dimensionality of the condition (e.g. speaker embedding). Set to 0 to ignore the condition.
        """
        pass

    def forward(
        self,
        features: torch.Tensor,
        lenghts: Optional[torch.Tensor],
        condition: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        inputs:
            features: [batch_size, channels, max_length], dtype=float
            lengths: [batch_size], dtype=float
            condition: [batch_size, 1, condition_dim] dtype=float
        returns:
            z: variational encoded features [batch_size, channels, max_length], dtype=float
            mean: [batch_size, channels, max_length], dtype=float
            log_s: log-scaled standard deviation, [batch_size, channels, max_length], dtype=float
            mask: [batch_size, 1, max_length], 1=not masked, 0=masked
        """
        raise NotImplementedError


class TextToSpeech(Protocol):
    pass


class GanTextToSpeech(TextToSpeech):
    pass


class GanTextToSpeechGenerator(Protocol):
    def forward(self, *args, **kwargs):
        pass

    def infer_tts(self, *args, **kwargs) -> torch.Tensor:
        pass


class DurationPredictor(Protocol):
    pass


class DurationDiscriminator(Protocol):
    pass


__all__ = [
    "Upsampler",
    "TextEncoder",
    "VariationalTextEncoder",
    "AcousticFeatureEncoder",
    "VariationalAcousticFeatureEncoder",
    "DurationPredictor",
    "DurationDiscriminator",
    "GanTextToSpeech",
    "GanTextToSpeechGenerator",
]

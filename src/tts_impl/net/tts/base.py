from typing import Optional, Protocol

import torch


class Upsampler(Protocol):
    """
    protocol for upsampling encoded text by duration

    inputs:
        hs: [BatchSize, Channels, MaxLength]
        ds: [BatchSize, MaxLength]
        h_masks: [BatchSize, 1, MaxLength]
        d_masks: [BatchSize, 1, MaxLength]
    outputs:
        features: [BatchSize, Channels, Length_Upsampled]
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
    inputs:
        phonemes_ids: [BatchSize, MaxLength], dtype=int
        lengths: [BatchSize], dtype=float
        g: [BatchSize, 1, SpeakerEmbeddingdim] dtype=float
    returns:
        z: encoded features [BatchSize, Channels, MaxLength], dtype=float
        mask: [BatchSize, 1, MaxLength], 1=not masked, 0=masked
    """

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        lenghts: Optional[torch.Tensor],
        g: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class AcousticFeatureEncoder(Protocol):
    """
    inputs:
        features: [BatchSize, Channels, MaxLength], dtype=float
        lengths: [BatchSize], dtype=float
        g: [BatchSize, 1, SpeakerEmbeddingdim] dtype=float
    returns:
        z: encoded features [BatchSize, Channels, MaxLength], dtype=float
        mask: [BatchSize, 1, MaxLength], 1=not masked, 0=masked
    """

    def forward(
        self,
        features: torch.Tensor,
        lenghts: Optional[torch.Tensor],
        g: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class VariationalTextEncoder(Protocol):
    """
    inputs:
        phonemes: [BatchSize, MaxLength], dtype=int
        lengths: [BatchSize], dtype=float
        g: [BatchSize, 1, SpeakerEmbeddingdim] dtype=float
    returns:
        z: variational encoded features [BatchSize, Channels, MaxLength], dtype=float
        mean: [BatchSize, Channels, MaxLength], dtype=float
        log_s: log-scaled standard deviation, [BatchSize, Channels, MaxLength], dtype=float
        mask: [BatchSize, 1, MaxLength], 1=not masked, 0=masked
    """

    def forward(
        self,
        features: torch.Tensor,
        lenghts: Optional[torch.Tensor],
        g: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class VariationalAcousticFeatureEncoder(
    Protocol,
):
    """
    inputs:
        features: [BatchSize, Channels, MaxLength], dtype=float
        lengths: [BatchSize], dtype=float
        g: [BatchSize, 1, SpeakerEmbeddingdim] dtype=float
    returns:
        z: variational encoded features [BatchSize, Channels, MaxLength], dtype=float
        mean: [BatchSize, Channels, MaxLength], dtype=float
        log_s: log-scaled standard deviation, [BatchSize, Channels, MaxLength], dtype=float
        mask: [BatchSize, 1, MaxLength], 1=not masked, 0=masked
    """

    def forward(
        self,
        features: torch.Tensor,
        lenghts: Optional[torch.Tensor],
        g: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

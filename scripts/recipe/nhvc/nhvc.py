import torch
from lightning import LightningDataModule

from tts_impl.net.vc.nhvc import NhvcLightningModule
from tts_impl.utils.datamodule import VcDataModule
from tts_impl.utils.preprocess import (
    Mixdown,
    PitchEstimation,
    Preprocessor,
    VcCacheWriter,
    VcDataCollector,
)
from tts_impl.utils.recipe import Recipe


class Nhvc(Recipe):
    """
    NHVC (Neural Homomorphic Voice Conversion) Recipe

    This recipe implements voice conversion using:
    - HuBERT-based phoneme embedding distillation
    - Pitch estimation and conversion
    - HomomorphicVocoder for waveform synthesis
    - Adversarial training with HiFiGAN discriminator
    """

    def __init__(self):
        super().__init__(NhvcLightningModule, "nhvc")

    def preprocess(
        self,
        target_dir: str = "your_target_dir",
        sample_rate: int = 24000,
        num_frames: int = 500,
        frame_size: int = 256,
        dataset_cache_path: str = "dataset_cache",
    ):
        """
        Preprocess audio data for NHVC training.

        Args:
            target_dir: Directory containing training data organized by speaker
            sample_rate: Target sample rate for audio (default: 48000)
            num_frames: Number of frames per sample (default: 500)
            frame_size: Hop length for frame extraction (default: 256)
            dataset_cache_path: Path to save preprocessed dataset cache

        Note:
            Mel spectrogram extraction is performed during training, not preprocessing.
        """
        preprocess = Preprocessor()

        # Collect audio data from target directory
        preprocess.with_collector(
            VcDataCollector(
                target_dir,
                sample_rate=sample_rate,
                max_length=frame_size * num_frames,
            )
        )

        # Mixdown to mono
        preprocess.with_extractor(Mixdown())

        # Pitch extraction
        preprocess.with_extractor(
            PitchEstimation(
                frame_size=frame_size,
                algorithm="yin",
            )
        )

        # Write to cache
        preprocess.with_writer(VcCacheWriter(dataset_cache_path))

        # Run preprocessing pipeline
        preprocess.run()

    def prepare_datamodule(
        self,
        root_dir: str = "dataset_cache",
        batch_size: int = 16,
        frame_size: int = 256,
        num_frames: int = 500,
    ) -> LightningDataModule:
        """
        Prepare data module for training.

        Args:
            root_dir: Root directory of preprocessed dataset cache
            batch_size: Batch size for training (default: 16)
            frame_size: Hop length used in preprocessing (default: 256)
            num_frames: Number of frames per sample (default: 500)

        Returns:
            LightningDataModule configured for NHVC training

        Note:
            Mel spectrogram is computed on-the-fly during training from waveform.
        """
        datamodule = VcDataModule(
            root=root_dir,
            batch_size=batch_size,
            num_workers=1,
            sizes={
                "f0": num_frames,
                "waveform": num_frames * frame_size,
            },
        )
        return datamodule


if __name__ == "__main__":
    Nhvc().cli()

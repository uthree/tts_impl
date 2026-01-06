from lightning import LightningDataModule

from tts_impl.net.vocoder.nsf_hifigan import NsfhifiganLightningModule
from tts_impl.utils.datamodule import AudioDataModule
from tts_impl.utils.preprocess import (
    AudioCacheWriter,
    AudioDataCollector,
    Mixdown,
    PitchEstimation,
    Preprocessor,
)
from tts_impl.utils.recipe import Recipe


class NsfHifigan(Recipe):
    def __init__(self):
        super().__init__(NsfhifiganLightningModule, "nsf_hifigan")

    def preprocess(
        self,
        target_dir: str = "your_target_dir",
        sample_rate: int = 22050,
        frame_size: int = 256,
        transcriptions_filename: str = "transcripts_utf8.txt",
        num_frames: int = 32,
    ):
        preprocess = Preprocessor()
        preprocess.with_collector(
            AudioDataCollector(
                target_dir, sample_rate=sample_rate, length=frame_size * num_frames
            )
        )
        # mixdown
        preprocess.with_extractor(Mixdown())
        preprocess.with_extractor(
            PitchEstimation(frame_size=frame_size, algorithm="fcpe")
        )
        preprocess.with_writer(AudioCacheWriter("dataset_cache"))
        preprocess.run()

    def prepare_datamodule(
        self,
        root_dir: str = "dataset_cache",
        batch_size: int = 16,
        frame_size: int = 256,
        num_frames: int = 32,
    ) -> LightningDataModule:
        datamodule = AudioDataModule(
            root=root_dir,
            batch_size=batch_size,
            num_workers=1,
            sizes={"waveform": frame_size * num_frames, "f0": num_frames},
        )
        return datamodule


if __name__ == "__main__":
    NsfHifigan().cli()

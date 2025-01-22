from lightning import LightningDataModule
from tts_impl.net.vocoder.ddsp.ultralight import UltraLightweightVocoderLightningModule
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
        super().__init__(UltraLightweightVocoderLightningModule, "ulvcdr")

    def preprocess(
        self,
        target_dir: str = "your_target_dir",
        sample_rate: int = 24000,
        length: int = 32768,
    ):
        preprocess = Preprocessor()
        preprocess.with_collector(
            AudioDataCollector(target_dir, sample_rate=sample_rate, length=length)
        )
        # mixdown
        preprocess.with_extractor(Mixdown())
        preprocess.with_extractor(PitchEstimation(frame_size=128, algorithm="fcpe"))
        preprocess.with_writer(AudioCacheWriter("dataset_cache"))
        preprocess.run()

    def prepare_datamodule(
        self, root_dir: str = "dataset_cache", batch_size: int = 16
    ) -> LightningDataModule:
        datamodule = AudioDataModule(
            root=root_dir, batch_size=batch_size, num_workers=1
        )
        return datamodule


if __name__ == "__main__":
    NsfHifigan().cli()

from lightning import LightningDataModule
from tts_impl.net.vocoder.ddsp import DdspVocoderLightningModule
from tts_impl.utils.datamodule import AudioDataModule
from tts_impl.utils.preprocess import (
    VcCacheWriter,
    VcDataCollector,
    Mixdown,
    PitchEstimation,
    Preprocessor,
)
from tts_impl.utils.recipe import Recipe
import torch


class NsfHifigan(Recipe):
    def __init__(self):
        super().__init__(DdspVocoderLightningModule, "ddsp")

    def preprocess(
        self,
        target_dir: str = "your_target_dir",
        sample_rate: int = 24000,
        length: int = 24000*5,
        frame_size: int = 256,
        dataset_cache_path: str = "dataset_cache"
    ):
        preprocess = Preprocessor()
        preprocess.with_collector(
            VcDataCollector(target_dir, sample_rate=sample_rate, max_length=length)
        )
        # mixdown
        preprocess.with_extractor(Mixdown())
        preprocess.with_extractor(
            PitchEstimation(frame_size=frame_size, algorithm="fcpe", device=torch.device("cuda" if torch.cuda.is_available else "cpu"))
        )
        preprocess.with_writer(VcCacheWriter(dataset_cache_path))
        preprocess.run()

    def prepare_datamodule(
        self, root_dir: str = "dataset_cache", batch_size: int = 4
    ) -> LightningDataModule:
        datamodule = AudioDataModule(
            root=root_dir, batch_size=batch_size, num_workers=1
        )
        return datamodule


if __name__ == "__main__":
    NsfHifigan().cli()

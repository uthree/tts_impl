from lightning import LightningDataModule
from tts_impl.net.vc.ddsp_vc.lightning import DdspVcLightningModule
from tts_impl.utils.datamodule import VcDataModule
from tts_impl.utils.preprocess import (
    VcDataCollector,
    VcCacheWriter,
    Mixdown,
    PitchEstimation,
    Preprocessor,
)
from tts_impl.utils.recipe import Recipe
import torch

class DdspVc(Recipe):
    def __init__(self):
        super().__init__(DdspVcLightningModule, "ddsp_vc")

    def preprocess(
        self,
        target_dir: str = "your_target_dir",
        sample_rate: int = 24000,
        max_length: int = 24000*5
    ):
        preprocess = Preprocessor()
        preprocess.with_collector(
            VcDataCollector(target_dir, sample_rate=sample_rate, max_length=max_length)
        )
        # mixdown
        preprocess.with_extractor(Mixdown())
        preprocess.with_extractor(PitchEstimation(frame_size=256, algorithm="fcpe", device=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))))
        preprocess.with_writer(VcCacheWriter("dataset_cache"))
        preprocess.run()

    def prepare_datamodule(
        self, root_dir: str = "dataset_cache", batch_size: int = 16, max_length: int = 24000*5
    ) -> LightningDataModule:
        datamodule = VcDataModule(
            root=root_dir, batch_size=batch_size, num_workers=1, sizes={"waveform": max_length}
        )
        return datamodule


if __name__ == "__main__":
    DdspVc().cli()

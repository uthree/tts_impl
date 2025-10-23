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


class NsfHifigan(Recipe):
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
        preprocess = Preprocessor()
        preprocess.with_collector(
            VcDataCollector(
                target_dir, sample_rate=sample_rate, max_length=frame_size * num_frames
            )
        )
        # mixdown
        preprocess.with_extractor(Mixdown())
        # pitch extraction
        preprocess.with_extractor(
            PitchEstimation(
                frame_size=frame_size,
                algorithm="fcpe",
                device=torch.device("cuda" if torch.cuda.is_available else "cpu"),
            )
        )
        preprocess.with_writer(VcCacheWriter(dataset_cache_path))
        preprocess.run()

    def prepare_datamodule(
        self,
        root_dir: str = "dataset_cache",
        batch_size: int = 16,
        frame_size: int = 256,
        num_frames: int = 500,
    ) -> LightningDataModule:
        datamodule = VcDataModule(
            root=root_dir,
            batch_size=batch_size,
            num_workers=1,
            sizes={"f0": num_frames, "waveform": num_frames * frame_size},
        )
        return datamodule


if __name__ == "__main__":
    NsfHifigan().cli()

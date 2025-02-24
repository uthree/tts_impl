from lightning import LightningDataModule
from tts_impl.net.tts.vits import VitsLightningModule
from tts_impl.utils.datamodule import AudioDataModule
from tts_impl.utils.preprocess import (
    TTSCacheWriter,
    TTSDataCollector,
    Mixdown,
    G2PExtractor,
    WaveformLengthExtractor,
    Preprocessor,
)
from tts_impl.utils.recipe import Recipe
from tts_impl.g2p import Grapheme2Phoneme
from tts_impl.g2p.pyopenjtalk import PyopenjtalkG2P


class Vits(Recipe):
    def __init__(self):
        super().__init__(VitsLightningModule, "vits")

    def preprocess(
        self,
        target_dir: str = "your_target_dir",
        sample_rate: int = 24000,
    ):
        preprocess = Preprocessor()
        g2p = Grapheme2Phoneme({"ja": PyopenjtalkG2P()})
        preprocess.with_collector(
            TTSDataCollector(target_dir, sample_rate=sample_rate, language="ja", transcriptions_filename="transcripts_utf8.txt")
        )
        preprocess.with_extractor(Mixdown())
        preprocess.with_extractor(
            G2PExtractor(
                g2p,
            )
        )
        preprocess.with_extractor(
            WaveformLengthExtractor(frame_size=256, max_frames=1000)
        )
        preprocess.with_writer(TTSCacheWriter("dataset_cache"))
        preprocess.run()

    def prepare_datamodule(
        self, root_dir: str = "dataset_cache", batch_size: int = 2
    ) -> LightningDataModule:
        datamodule = AudioDataModule(
            root=root_dir,
            batch_size=batch_size,
            num_workers=1,
            sizes={"waveform": 256000},
        )
        return datamodule


if __name__ == "__main__":
    Vits().cli()

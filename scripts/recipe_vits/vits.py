from lightning import LightningDataModule
from tts_impl.g2p import Grapheme2Phoneme
from tts_impl.g2p.pyopenjtalk import PyopenjtalkG2P
from tts_impl.net.tts.vits import VitsLightningModule
from tts_impl.utils.datamodule import TTSDataModule
from tts_impl.utils.preprocess import (
    G2PExtractor,
    Mixdown,
    Preprocessor,
    TTSCacheWriter,
    TTSDataCollector,
    WaveformLengthExtractor,
)
from tts_impl.utils.recipe import Recipe
import torch
import torchaudio


class Vits(Recipe):
    def __init__(self):
        super().__init__(VitsLightningModule, "vits")

    def preprocess(
        self,
        target_dir: str = "your_target_dir",
        sample_rate: int = 22050,
        transcriptions_filename: str = "transcripts_utf8.txt",
    ):
        preprocess = Preprocessor()
        g2p = Grapheme2Phoneme({"ja": PyopenjtalkG2P()})
        preprocess.with_collector(
            TTSDataCollector(
                target_dir,
                sample_rate=sample_rate,
                language="ja",
                transcriptions_filename=transcriptions_filename,
            )
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
        datamodule = TTSDataModule(
            root=root_dir,
            batch_size=batch_size,
            num_workers=1,
            sizes={"waveform": 256000},
        )
        return datamodule

    def infer(self, text: str):
        with torch.inference_mode():
            model = self.load_model()
            gen = model.generator
            g2p = Grapheme2Phoneme({"ja": PyopenjtalkG2P()})
            x, x_lengths, _ = g2p.encode([text], ["ja"])
            wf = gen.infer(x, x_lengths, sid=torch.LongTensor([0])).squeeze(1)
            torchaudio.save("output.wav", wf, 22050)


if __name__ == "__main__":
    Vits().cli()

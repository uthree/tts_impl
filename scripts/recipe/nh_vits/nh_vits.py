from pathlib import Path

import torch
import torchaudio
from lightning import LightningDataModule
from tts_impl.g2p import Grapheme2Phoneme
from tts_impl.g2p.pyopenjtalk import PyopenjtalkG2P
from tts_impl.net.tts.nh_vits import NhvitsLightningModule
from tts_impl.utils.datamodule import TTSDataModule
from tts_impl.utils.preprocess import (
    G2PExtractor,
    Mixdown,
    PitchEstimation,
    Preprocessor,
    TTSCacheWriter,
    TTSDataCollector,
    WaveformLengthExtractor,
)
from tts_impl.utils.recipe import Recipe


class Nhvits(Recipe):
    def __init__(self):
        super().__init__(NhvitsLightningModule, "nh_vits")

    def preprocess(
        self,
        target_dir: str = "your_target_dir",
        sample_rate: int = 48000,
        transcriptions_filename: str = "transcripts_utf8.txt",
        frame_size: int = 256,
        pe_algorithm: str = "fcpe",
        num_frames: int = 1000,
    ):
        preprocess = Preprocessor()
        g2p = Grapheme2Phoneme({"ja": PyopenjtalkG2P()})
        preprocess.with_collector(
            TTSDataCollector(
                target_dir,
                sample_rate=sample_rate,
                language="ja",
                transcriptions_filename=transcriptions_filename,
                concatenate=True,
                max_length=256000,
                filename_blacklist=["falset", "whisper"],
            )
        )
        preprocess.with_extractor(Mixdown())
        preprocess.with_extractor(
            G2PExtractor(
                g2p,
            ),
        )
        preprocess.with_extractor(
            PitchEstimation(
                frame_size,
                pe_algorithm,
                device=(
                    torch.device("cuda")
                    if torch.cuda.is_available
                    else torch.device("cpu")
                ),
            )
        )
        preprocess.with_extractor(
            WaveformLengthExtractor(frame_size=frame_size, max_frames=num_frames)
        )
        preprocess.with_writer(TTSCacheWriter("dataset_cache"))
        preprocess.run()

    def prepare_datamodule(
        self,
        root_dir: str = "dataset_cache",
        batch_size: int = 16,
        frame_size: int = 256,
        num_frames: int = 1000,
    ) -> LightningDataModule:
        datamodule = TTSDataModule(
            root=root_dir,
            batch_size=batch_size,
            num_workers=1,
            sizes={"waveform": frame_size * num_frames, "f0": num_frames},
        )
        return datamodule

    def infer(self, text: str = "", sid: int = 0):
        with torch.inference_mode():
            outputs_dir = Path("outputs")
            gen = self.model.generator
            g2p = Grapheme2Phoneme({"ja": PyopenjtalkG2P()})
            x, x_lengths, _ = g2p.encode([text], ["ja"])
            wf = gen.infer(x, x_lengths, sid=torch.LongTensor([sid])).squeeze(1)
            outputs_dir.mkdir(exist_ok=True)
            torchaudio.save(outputs_dir / f"{sid}.wav", wf, 22050)


if __name__ == "__main__":
    Nhvits().cli()

import fire
import torch
from tts_impl.utils.preprocess import (
    AudioCacheWriter,
    AudioDataCollector,
    FunctionalExtractor,
    Mixdown,
    Preprocessor,
    PitchEstimation
)


def run_preprocess(target_dir: str):
    preprocess = Preprocessor()
    preprocess.with_collector(
        AudioDataCollector(target_dir, sample_rate=22050, length=8192)
    )
    # mixdown
    preprocess.with_extractor(Mixdown())
    preprocess.with_extractor(PitchEstimation(frame_size=256))
    preprocess.with_writer(AudioCacheWriter())
    preprocess.run()


if __name__ == "__main__":
    fire.Fire(run_preprocess)

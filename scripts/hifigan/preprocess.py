import fire
from tts_impl.utils.preprocess import AudioCacheWriter, AudioDataCollector, Preprocessor, FunctionalExtractor

import torch


def run_preprocess(target_dir: str):
    preprocess = Preprocessor()
    preprocess.with_collector(
        AudioDataCollector(target_dir, sample_rate=22050, length=8192)
    )
    # mixdown
    preprocess.with_extractor(FunctionalExtractor("waveform", "waveform", lambda x: x.sum(dim=0, keepdim=True)))
    preprocess.with_writer(AudioCacheWriter())
    preprocess.run()


if __name__ == "__main__":
    fire.Fire(run_preprocess)

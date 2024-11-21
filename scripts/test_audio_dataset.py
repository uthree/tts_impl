from tts_impl.preprocess import Preprocessor, AudioDataCollector, AudioCacheWriter, FunctionalExtractor
from tts_impl.transforms import LogMelSpectrogram
import torch.nn as nn

preprocess = Preprocessor()
preprocess.with_collector(AudioDataCollector("D:\datasets\jvs_experimental", length=48000, sample_rate=48000))
ext_fn_mel = LogMelSpectrogram(48000, 1920, 480, center=True)
preprocess.with_extractor(FunctionalExtractor("waveform", "spectrogram", ext_fn_mel))
preprocess.with_writer(AudioCacheWriter())
preprocess.run()
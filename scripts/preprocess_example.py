from tts_impl.preprocess import Preprocessor, AudioDataCollector, AudioCacheWriter, FunctionalExtractor
from tts_impl.transforms import LogMelSpectrogram
import torch.nn as nn

# Initialize "Preprocessor"
preprocess = Preprocessor()

# add Collector
preprocess.with_collector(AudioDataCollector("D:\datasets\jvs_experimental", length=65536, sample_rate=22050))

# add extractor
ext_fn_mel = LogMelSpectrogram()
preprocess.with_extractor(FunctionalExtractor("waveform", "spectrogram", ext_fn_mel))

# set cache writer
preprocess.with_writer(AudioCacheWriter())

# run preprocess
preprocess.run()
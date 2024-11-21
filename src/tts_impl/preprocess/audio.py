import os
from pathlib import Path
from typing import List, Optional, Union

import torch
import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm

from tts_impl.functional import adjust_size

from .base import CacheWriter, DataCollector


class AudioDataCollector(DataCollector):
    def __init__(
        self,
        target: Union[str, os.PathLike],
        length: Optional[int] = None,
        formats: List[str] = ["wav", "mp3", "flac", "ogg"],
        sample_rate: Optional[int] = None,
    ):
        """
        Args:
            target: Target directory paths.
            length: Waveform lengths (number of samples), If given, format to that length.
            formats: Target file extensions
            sample_rate: If given, resampling will be performed.
        """
        self.target = Path(target)
        self.length = length
        self.formats = formats
        self.sample_rate = sample_rate

    def __iter__(self):
        # collect paths
        audio_file_paths = []

        tqdm.write(f"Collecting audio files in {self.target} ...")
        for fmt in self.formats:
            tqdm.write(f"scanning format: {fmt}")
            for path in self.target.glob(f"**/*.{fmt}"):
                audio_file_paths.append(path)

        tqdm.write(f"Collected {len(audio_file_paths)} file(s).")

        for path in audio_file_paths:
            tqdm.write(f"loading {path} ...")
            wf, orig_sr = torchaudio.load(path)
            # wf: [C, L]

            if self.sample_rate is not None:
                tqdm.write(f"resampling {orig_sr}Hz to {self.sample_rate}Hz ...")
                sr = orig_sr
                if orig_sr != self.sample_rate:
                    wf = resample(wf, orig_sr, self.sample_rate)
                    sr = self.sample_rate

            if self.length is None:
                yield {"waveform": wf, "sample_rate": sr}
            else:
                # split data
                chunks = torch.split(wf, self.length, dim=1)
                for chunk in chunks:
                    chunk = adjust_size(chunk, self.length)
                    yield {"waveform": chunk, "sample_rate": sr}


class AudioCacheWriter(CacheWriter):
    def __init__(
        self, cache_dir: Union[str, os.PathLike] = "dataset_cache", format: str = "flac"
    ):
        super().__init__(cache_dir)
        self.format = format
        self.counter = 0

    def write(self, data: dict):
        wf = data.pop("waveform")
        sr = data["sample_rate"]
        torchaudio.save(self.cache_dir / f"{self.counter}.{self.format}", wf, sr)
        torch.save(data, self.cache_dir / f"{self.counter}.pt")
        self.counter += 1
import os
from pathlib import Path
from typing import List, Literal, Optional, Union

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

        Warning: You cannot load a file that is longer than the memory capacity of your computer. If the audio file is too long, please split it in advance.
        """
        self.target = Path(target)
        self.length = length
        self.formats = formats
        self.sample_rate = sample_rate

    def __iter__(self):
        # collect paths
        audio_file_paths = []

        tqdm.write(f"Collecting audio files from {self.target} ...")
        for fmt in self.formats:
            tqdm.write(f"Scanning format: {fmt}")
            for path in self.target.glob(f"**/*.{fmt}"):
                audio_file_paths.append(path)

        tqdm.write(f"Collected {len(audio_file_paths)} file(s).")

        # yield loop with tqdm progress bar
        for path in tqdm(
            audio_file_paths, desc=f"Collecting audio files from {self.target}"
        ):
            tqdm.write(f"Loading {path} ...")
            wf, orig_sr = torchaudio.load(path)
            # wf: [C, L]

            # resampe
            if self.sample_rate is not None:
                sr = orig_sr
                if orig_sr != self.sample_rate:
                    tqdm.write(f"Resampling {orig_sr}Hz to {self.sample_rate}Hz ...")
                    wf = resample(wf, orig_sr, self.sample_rate)
                    sr = self.sample_rate

            # split and 'yield'
            if self.length is None:
                yield {"waveform": wf, "sample_rate": sr}
            else:
                tqdm.write("Splitting ...")
                # split data
                chunks = torch.split(wf, self.length, dim=1)
                tqdm.write(f"Got {len(chunks)} audio clip(s).")
                for chunk in chunks:
                    chunk = adjust_size(chunk, self.length)
                    yield {"waveform": chunk, "sample_rate": sr}


class AudioCacheWriter(CacheWriter):
    def __init__(
        self,
        cache_dir: Union[str, os.PathLike] = "dataset_cache",
        format: Literal["flac", "wav", "mp3", "ogg"] = "flac",
        delete_old_cache=True,
    ):
        """
        Args:
            cache_dir: The dataset cache directory. default: `"dataset_cache"`
            format: Audio file extensions, default: `"flac"`
        """
        super().__init__(cache_dir, delete_old_cache=delete_old_cache)
        self.format = format
        self.counter = 0

    def write(self, data: dict):
        wf = data.pop("waveform")
        sr = data["sample_rate"]
        torchaudio.save(self.cache_dir / f"{self.counter}.{self.format}", wf, sr)
        torch.save(data, self.cache_dir / f"{self.counter}.pt")
        self.counter += 1

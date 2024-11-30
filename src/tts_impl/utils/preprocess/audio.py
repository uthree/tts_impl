import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import torch
import torchaudio
from rich.progress import track
from torchaudio.functional import resample
from tts_impl.functional import adjust_size, estimate_f0

from .base import CacheWriter, DataCollector, Extractor, FunctionalExtractor


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

        for fmt in self.formats:
            for path in self.target.glob(f"**/*.{fmt}"):
                audio_file_paths.append(path)

        # yield loop with progress bar
        self.logger.info(f"collecting from {self.target}")
        for path in track(audio_file_paths, console=self.console):
            wf, orig_sr = torchaudio.load(path)
            # wf: [C, L]

            # resampe
            if self.sample_rate is not None:
                sr = orig_sr
                if orig_sr != self.sample_rate:
                    wf = resample(wf, orig_sr, self.sample_rate)
                    sr = self.sample_rate

            # split and 'yield'
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
        self,
        root: Union[str, os.PathLike] = "dataset_cache",
        format: Literal["flac", "wav", "mp3", "ogg"] = "flac",
        delete_old_cache: bool = True,
        max_files_per_dir: int = 10000,
    ):
        """
        Args:
            cache_dir: The dataset cache directory. default: `"dataset_cache"`
            format: Audio file extensions, default: `"flac"`
        """
        super().__init__()
        self.root = Path(root)
        self.format = format
        self.max_files_per_dir = max_files_per_dir
        self.delete_old_cache = delete_old_cache

        self.counter = 0
        self.dir_counter = 0
        self.sample_rate = None

    def prepare(self):
        self.counter = 0
        self.dir_counter = 0
        self.root.mkdir(parents=True, exist_ok=True)

        if self.delete_old_cache:
            if self.root.exists():
                shutil.rmtree(self.root)
                self.logger.log(logging.INFO, f"Deleted cache directory: {self.root}")

    def write(self, data: dict):
        wf = data.pop("waveform")
        sr = data["sample_rate"]
        self.sample_rate = sr
        subdir = self.root / f"{self.dir_counter}"
        subdir.mkdir(parents=True, exist_ok=True)
        torchaudio.save(subdir / f"{self.counter}.{self.format}", wf, sr)
        torch.save(data, subdir / f"{self.counter}.pt")
        self.counter += 1
        if self.counter % self.max_files_per_dir == 0:
            self.dir_counter += 1

    def finalize(self):
        metadata = dict()
        if self.sample_rate is not None:
            metadata["sample_rate"] = self.sample_rate
        with open(self.root / "metadata.json", mode="w") as f:
            json.dump(metadata, f)


class Mixdown(FunctionalExtractor):
    def __init__(self, dim=0):
        super().__init__("waveform", "waveform", lambda x: x.sum(dim, keepdim=True))


class PitchEstimation(Extractor):
    def __init__(
        self,
        frame_size: int,
        algorithm: Literal["harvest", "dio", "fcpe", "yin"] = "harvest",
    ):
        super().__init__()
        self.algorithm = algorithm
        self.frame_size = frame_size

    def extract(self, data: dict[str, Any]) -> dict[str, Any]:
        wf = data["waveform"]
        sr = data["sample_rate"]

        f0 = estimate_f0(
            wf.unsqueeze(0), sr, self.frame_size, algorithm=self.algorithm
        ).squeeze(0)
        data["f0"] = f0

        return data

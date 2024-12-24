import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Mapping, Optional, Union

import torch
import torchaudio
from rich.progress import track
from torchaudio.functional import resample
from tts_impl.functional import adjust_size, estimate_f0

from .base import CacheWriter, DataCollector, Extractor, FunctionalExtractor


class TTSDataCollector(DataCollector):
    def __init__(
        self,
        target: Union[str, os.PathLike],
        formats: List[str] = ["wav", "mp3", "flac", "ogg"],
        sample_rate: Optional[int] = None,
    ):
        self.target = Path(target)
        self.formats = formats
        self.sample_rate = sample_rate

    def __iter__(self) -> Generator[Mapping[str, Any]]:
        subdirs = [d for d in self.target.iterdir() if d.is_dir()]
        for subdir in subdirs:
            generator = self.process_subdir(subdir)
            for data in generator:
                yield data

    def process_subdir(self, subdir: Path) -> Generator[Mapping[str, Any]]:
        """
        Override as needed.
        """
        speaker = subdir.name
        transcriptions_path = subdir / "transcriptions.txt"
        with open(transcriptions_path) as f:
            transcriptions = f.read().split("\n")
        for transciption in transcriptions:
            transcription = transciption.split(":")
            if len(transciption) == 2:
                fname = transcription[0]
                trans = transcription[1]
                for fmt in self.formats:
                    audio_path = subdir / f"{fname}.{fmt}"
                    if audio_path.exists():
                        wf, sr = torchaudio.load(audio_path)
                        data = {
                            "speaker": speaker,
                            "waveform": wf,
                            "sample_rate": sr,
                            "transcription": trans
                        }
                        yield data
                        break

class TTSCacheWriter(CacheWriter):
    def __init__(
        self,
        root: Union[str, os.PathLike] = "dataset_cache",
        format: Literal["flac", "wav", "mp3", "ogg"] = "flac",
        delete_old_cache: bool = True,
    ):
        self.sample_rate = None
        self.root = Path(root)
        self.format = format
        self.delete_old_cache = delete_old_cache
        self.counter = dict()
        super().__init__()

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True)

        if self.delete_old_cache:
            if self.root.exists():
                shutil.rmtree(self.root)
                self.logger.log(logging.INFO, f"Deleted cache directory: {self.root}")

    def write(self, data: dict[str, Any]):
        wf = data.pop("")
        speaker = data["speaker"]
        sr = data["sample_rate"]
        self.sample_rate = sr
        subdir = self.root / f"{speaker}"

        if speaker not in self.counter:
            self.counter[speaker] = 0
        else:
            self.counter[speaker] += 1

        subdir.mkdir(parents=True, exist_ok=True)
        counter = self.counter[speaker]
        torchaudio.save(subdir / f"{counter}.{self.format}", wf, sr)
        torch.save(data, subdir / f"{counter}.pt")

    def finalize(self):
        metadata = dict()
        speakers = sorted(self.counter.keys())
        metadata["speakrs"] = speakers
        if self.sample_rate is not None:
            metadata["sample_rate"] = self.sample_rate
        with open(self.root / "metadata.json", mode="w") as f:
            json.dump(metadata, f)

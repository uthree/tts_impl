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
from tts_impl.g2p import Grapheme2Phoneme
import re

from .base import CacheWriter, DataCollector, Extractor, FunctionalExtractor


class TTSDataCollector(DataCollector):
    def __init__(
        self,
        target: Union[str, os.PathLike],
        formats: List[str] = ["wav", "mp3", "flac", "ogg"],
        sample_rate: Optional[int] = None,
        language: Optional[str] = None,
        transcriptions_filename: str = "transcriptions.txt",
        transcriptions_encoding:str = "utf-8"
    ):
        self.target = Path(target)
        self.formats = formats
        self.sample_rate = sample_rate
        self.language = language
        self.transcriptions_filename = transcriptions_filename
        self.transcriptions_encoding=transcriptions_encoding

    def __iter__(self) -> Generator[Mapping[str, Any], None, None]:
        subdirs = [d for d in self.target.iterdir() if d.is_dir()]
        for subdir in subdirs:
            generator = self.process_subdir(subdir)
            for data in generator:
                yield data


    def load_with_resample(self, path: Path) -> tuple[torch.Tensor, int]:
        wf, sr = torchaudio.load(path)
        if self.sample_rate is not None and sr != self.sample_rate:
            wf = resample(wf, sr, self.sample_rate)
            sr = self.sample_rate
        return wf, sr

    def parse_transcriptions(self, transcriptions: str) -> dict[str, dict[str]]:
        lines = transcriptions.split("\n")
        r = dict()
        for line in lines:
            m = re.match("(.+)\:(.+)", line)
            if m is not None:
                fname, trns = m.groups()
                r[fname] = {"name": fname, "transcription": trns}
        return r


    def process_subdir(self, subdir: Path) -> Generator[Mapping[str, Any], None, None]:
        speaker = subdir.name
        self.logger.info(f"Collecting speaker: `{speaker}` 's data from {subdir} ...")
        self.logger.info("Scanning transcriptions ...")
        transcriptions_paths = [subdir / self.transcriptions_filename]
        transcriptions_paths += [d for d in subdir.rglob(self.transcriptions_filename)]
        transcriptions = dict()
        for trns_path in transcriptions_paths:
            if trns_path.exists():
                self.logger.info(f"  Load: {trns_path}")
                with open(trns_path, encoding=self.transcriptions_encoding) as f:
                    transcriptions |= self.parse_transcriptions(f.read())
        self.logger.info(f"Detected {len(transcriptions)} transcription(s).")

        self.logger.info(f"Processing audio files ...")
        audio_paths = [p for p in subdir.rglob("*") if p.suffix.lstrip(".") in self.formats]

        for apath in audio_paths:
            self.logger.debug(f"Processing: {apath}")
            if apath.stem in transcriptions:
                wf, sr = self.load_with_resample(apath)
                data = {
                    "waveform": wf,
                    "transcription": transcriptions[apath.stem]["transcription"],
                    "speaker": speaker,
                    "sample_rate": sr,
                    "language": self.language
                }
                yield data

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
        wf = data.pop("waveform")
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
        with open(self.root / "metadata.json", mode="w+") as f:
            json.dump(metadata, f)


class G2PExtractor(Extractor):
    def __init__(self, g2p: Grapheme2Phoneme, length: int = 100):
        super().__init__()
        self.g2p = g2p
        self.length = length

    def extract(self, data: dict[str, Any]) -> dict[str, Any]:
        language = data["language"]
        transcription = data["transcription"]
        phonemes, phonemes_lengths, language = self.g2p.encode(
            [transcription], [language], length=self.length
        )
        data["language"] = language.squeeze(0)
        data["phonemes_lengths"] = phonemes_lengths.squeeze(0)
        data["phonemes"] = phonemes.squeeze(0)
        return data


class WaveformLengthExtractor(Extractor):
    def __init__(self, frame_size: 256, max_frames: int = 1024):
        super().__init__()
        self.frame_size = frame_size
        self.max_frames = max_frames

    def extract(self, data: dict[str, Any]) -> dict[str, Any]:
        wf = data["waveform"]
        wf_length = wf.shape[1]
        num_frames = wf_length // self.frame_size + 1
        num_frames = min(num_frames, self.max_frames)
        data["acoustic_features_lengths"] = num_frames
        return data

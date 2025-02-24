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

from .base import CacheWriter, DataCollector, Extractor, FunctionalExtractor


class TTSDataCollector(DataCollector):
    def __init__(
        self,
        target: Union[str, os.PathLike],
        formats: List[str] = ["wav", "mp3", "flac", "ogg"],
        sample_rate: Optional[int] = None,
        language: Optional[str] = None,
    ):
        self.target = Path(target)
        self.formats = formats
        self.sample_rate = sample_rate
        self.language = language

    def __iter__(self) -> Generator[Mapping[str, Any], None, None]:
        subdirs = [d for d in self.target.iterdir() if d.is_dir()]
        for subdir in subdirs:
            generator = self.process_subdir(subdir)
            for data in generator:
                yield data

    def process_subdir(self, subdir: Path) -> Generator[Mapping[str, Any], None, None]:
        """
        Override as needed.
        """
        speaker = subdir.name
        transcriptions_path = subdir / "transcriptions.txt"
        with open(transcriptions_path, encoding='utf8') as f:
            transcriptions = f.read().split("\n")
        for transcription in transcriptions:
            transcription = transcription.split(":")
            if len(transcription) == 2:
                fname = transcription[0]
                trans = transcription[1]
                for fmt in self.formats:
                    audio_path = subdir / f"{fname}.{fmt}"
                    if audio_path.exists():
                        wf, sr = torchaudio.load(audio_path)

                        # resample if different sample rate.
                        if self.sample_rate is not None and sr != self.sample_rate:
                            wf = torchaudio.functional.resample(wf, sr, self.sample_rate)
                            sr = self.sample_rate

                        data = {
                            "speaker": speaker,
                            "waveform": wf,
                            "sample_rate": sr,
                            "transcription": trans,
                        }
                        if self.language is not None:
                            data["language"] = self.language
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
        num_frames = max(num_frames, self.max_frames)
        data["acoustic_features_lengths"] = num_frames
        return data

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Generator, Literal, Mapping

import torch
import torchaudio
from rich.progress import track
from torchaudio.functional import resample
from tts_impl.functional import adjust_size, estimate_f0

from .base import CacheWriter, DataCollector, Extractor, FunctionalExtractor


class VcDataCollector(DataCollector):
    def __init__(
        self,
        target: str | os.PathLike,
        formats: list[str] = ["wav", "mp3", "flac", "ogg"],
        sample_rate: int | None = None,
        language: str | None = None,
        filename_blacklist: list[str] = [],
        max_length: int | None = None,
    ):
        self.target = Path(target)
        self.formats = formats
        self.sample_rate = sample_rate
        self.language = language
        self.max_length = max_length
        self.filename_blacklist = filename_blacklist

    def __iter__(self) -> Generator[Mapping[str, Any], None, None]:
        subdirs = [d for d in self.target.iterdir() if d.is_dir()]
        for subdir in subdirs:
            generator = self._process_subdir(subdir)
            for data in generator:
                yield data

    def load_with_resample(self, path: Path) -> tuple[torch.Tensor, int]:
        wf, sr = torchaudio.load(path)
        if self.sample_rate is not None and sr != self.sample_rate:
            wf = resample(wf, sr, self.sample_rate)
            sr = self.sample_rate
        if self.max_length is not None:
            if wf.shape[1] > self.max_length:
                wf = wf[:, : self.max_length]
        return wf, sr

    def _process_subdir(self, subdir: Path) -> Generator[Mapping[str, Any], None, None]:
        self.logger.info(f"processing subdir: {subdir}")
        speaker = subdir.name
        audio_paths = [
            p for p in subdir.rglob("*") if p.suffix.lstrip(".") in self.formats
        ]
        for apath in audio_paths:
            self.logger.debug(f"Processing: {apath}")
            wf, sr = self.load_with_resample(apath)
            data = {
                "waveform": wf,
                "speaker": speaker,
                "sample_rate": sr,
            }
            yield data


class VcCacheWriter(CacheWriter):
    def __init__(
        self,
        root: str | os.PathLike = "dataset_cache",
        format: Literal["flac", "wav", "mp3", "ogg"] = "flac",
        delete_old_cache: bool = True,
    ):
        self.sample_rate = None
        self.root = Path(root)
        self.format = format
        self.delete_old_cache = delete_old_cache
        self.counter = dict()
        self.f0_stats = dict()  # Store F0 statistics per speaker
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

        # Accumulate F0 statistics for average pitch calculation
        if "f0" in data and speaker not in self.f0_stats:
            self.f0_stats[speaker] = []

        if "f0" in data:
            f0 = data["f0"]
            # Filter out unvoiced/silence frames (F0 < 20Hz)
            # These are typically unvoiced consonants or silence regions
            # and should not be included in average pitch calculation
            voiced_f0 = f0[f0 > 20.0]
            if len(voiced_f0) > 0:
                self.f0_stats[speaker].extend(voiced_f0.tolist())

        subdir.mkdir(parents=True, exist_ok=True)
        counter = self.counter[speaker]
        torchaudio.save(subdir / f"{counter}.{self.format}", wf, sr)
        torch.save(data, subdir / f"{counter}.pt")

    def finalize(self):
        from tts_impl.functional.midi import freq2note

        metadata = dict()
        speakers = sorted(self.counter.keys())
        metadata["speakers"] = speakers
        if self.sample_rate is not None:
            metadata["sample_rate"] = self.sample_rate

        # Calculate average pitch per speaker in MIDI scale (logarithmic)
        speaker_avg_pitch = {}
        for speaker, f0_list in self.f0_stats.items():
            if len(f0_list) > 0:
                # Convert all F0 values to MIDI scale first (logarithmic scale)
                midi_list = [freq2note(f0) for f0 in f0_list]
                # Calculate average in MIDI scale (human perception is logarithmic)
                avg_midi = sum(midi_list) / len(midi_list)
                speaker_avg_pitch[speaker] = round(avg_midi, 2)

                # For logging, also calculate Hz average for reference
                avg_f0_hz = sum(f0_list) / len(f0_list)
                self.logger.log(
                    logging.INFO,
                    f"Speaker {speaker}: avg MIDI = {avg_midi:.2f} (linear avg F0 = {avg_f0_hz:.2f} Hz for reference)",
                )

        if speaker_avg_pitch:
            metadata["speaker_avg_pitch"] = speaker_avg_pitch

        with open(self.root / "metadata.json", mode="w+", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

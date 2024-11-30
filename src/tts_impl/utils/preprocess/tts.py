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


class TtsCacheWriter(CacheWriter):
    def __init__(
        self,
        root: Union[str, os.PathLike] = "dataset_cache",
        format: Literal["flac", "wav", "mp3", "ogg"] = "flac",
        delete_old_cache: bool = True,
    ):
        super().__init__()

    def write(self, data: dict):
        pass

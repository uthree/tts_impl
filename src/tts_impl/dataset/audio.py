import os
from pathlib import Path
from typing import List, Literal, Union

import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """
    dataset class for vocoder, codec, etc...
    """

    def __init__(
        self,
        root: Union[str, os.PathLike] = "dataset_cache",
        format: Literal["wav", "flac", "mp3", "ogg"] = "flac",
    ):
        super().__init__()
        self.root = Path(root)
        self.audio_file_paths = []
        self.feature_paths = []
        self.format = format

        for path in self.root.glob(f"*.{self.format}"):
            self.audio_file_paths.append(path)
            self.feature_paths.append(path.with_suffix(".pt"))

    def __getitem__(self, idx):
        wf, _ = torchaudio.load(self.audio_file_paths[idx])
        wf = wf.sum(dim=0)
        data = torch.load(self.feature_paths[idx])
        return wf, data

    def __len__(self):
        return len(self.audio_file_paths)

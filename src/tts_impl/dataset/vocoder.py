from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class VocoderDataset(Dataset):
    def __init__(self, cache_dir="dataset_cache"):
        super().__init__()
        self.root = Path(cache_dir)
        self.audio_file_paths = []
        self.feature_paths = []

        for path in self.root.glob("*.wav"):
            self.audio_file_paths.append(path)
            self.feature_paths.append(path.with_suffix(".pt"))

    def __getitem__(self, idx):
        wf, sr = torchaudio.load(self.audio_file_paths[idx])
        features = torch.load(self.feature_paths[idx])
        return wf.mean(dim=0), features

    def __len__(self):
        return len(self.audio_file_paths)

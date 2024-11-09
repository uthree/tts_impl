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
        wf, _ = torchaudio.load(self.audio_file_paths[idx])
        wf = wf.sum(dim=0)
        data = torch.load(self.feature_paths[idx])
        spec = data["spectrogram"]
        return wf, spec

    def __len__(self):
        return len(self.audio_file_paths)


class VocoderDatasetNsf(Dataset):
    def __init__(self, cache_dir="dataset_cache"):
        super().__init__()
        self.root = Path(cache_dir)
        self.audio_file_paths = []
        self.feature_paths = []

        for path in self.root.glob("*.wav"):
            self.audio_file_paths.append(path)
            self.feature_paths.append(path.with_suffix(".pt"))

    def __getitem__(self, idx):
        wf, _ = torchaudio.load(self.audio_file_paths[idx])
        wf = wf.sum(dim=0)
        data = torch.load(self.feature_paths[idx])
        spec = data["spectrogram"]
        f0 = data["f0"]
        return wf, spec, f0

    def __len__(self):
        return len(self.audio_file_paths)

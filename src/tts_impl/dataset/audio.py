import os
from pathlib import Path
from typing import Union, Dict, Optional

import torch
import torchaudio
from torch.utils.data import Dataset
from tts_impl.functional.pad import adjust_size_1d
from torchaudio.functional import resample


class AudioDataset(Dataset):
    """
    dataset class for vocoder, codec, etc...

    Args:
        root: PathLike, root directory path.
        format: str, audio file extension. default="frac"
        sample_rate: Option[int], output sampling rate
        lengths: Dict[str, int]
    
    details of lengths:
        Option to adjust the length of data.
        If the length exceeds the limit, the excess will be truncated, and if the length is insufficient, zero padding will be performed.
        The key is the name of the data, and the value is the length.
        Only supports tensors with (batch_size, channels, length).

        Applied to data stored in .pt files.
        The key for audio waveforms is "waveform".

        for example: `{"waveform": 48000, "f0": 100, "spectrogram": 100}`
    """

    def __init__(
        self,
        root: Union[str, os.PathLike] = "dataset_cache",
        format: str = "flac",
        sample_rate: Optional[int] =48000,
        lengths: Dict[str, int] = {},
    ):
        super().__init__()
        self.root = Path(root)
        self.audio_file_paths = []
        self.feature_paths = []
        self.format = format
        self.lengths = lengths
        self.sample_rate = sample_rate

        # get all paths
        for path in self.root.glob(f"*.{self.format}"):
            if os.path.exists(path.with_suffix(".pt")):
                self.audio_file_paths.append(path)
                self.feature_paths.append(path.with_suffix(".pt"))

    def __getitem__(self, idx):
        wf, sr = torchaudio.load(self.audio_file_paths[idx])

        # resample if different sample_rate
        if self.sample_rate is not None:
            if sr != self.sample_rate:
                wf = resample(wf, sr, self.sample_rate)

        # mix-down
        wf = wf.sum(dim=0, keepdim=True)

        # load other features
        data = torch.load(self.feature_paths[idx])

        data['waveform'] = wf

        # adjust length
        for k in self.lengths.keys:
            v = self.lengths[k]
            if k in data:
                if data[k].ndim == 3:
                    data[k] = adjust_size_1d(data[k], v)
        return data

    def __len__(self):
        return len(self.audio_file_paths)

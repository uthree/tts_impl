import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.functional import resample
from tts_impl.functional.pad import adjust_size


class AudioDataset(Dataset):
    """
    dataset class for vocoder, codec, etc...
    """

    def __init__(
        self,
        root: Union[str, os.PathLike] = "dataset_cache",
        format: str = "flac",
        sample_rate: Optional[int] = None,
        sizes: Dict[str, Union[int, Tuple[int], List[int]]] = {},
        mix_down: bool = False,
        weights_only=True,
    ):
        """
        Args:
            root: PathLike, root directory path.
            format: str, audio file extension. default="frac"
            sample_rate: Option[int], output sampling rate
            sizes: Dict[str, Union[int, Tuple[int], List[int]]]
            mix_down: bool, When True, it will be mixed down to mono (channels=1).
            weights_only: bool

        Details of lengths:
            Option to adjust the length of data.
            If the length exceeds the limit, the excess will be truncated, and if the length is insufficient, zero padding will be performed.
            The key is the name of the data, and the value is the size.

            Applied to data stored in .pt files.
            The key for audio waveforms is "waveform".

            for example: `{"waveform": 48000, "f0": 100, "spectrogram": 100}`
        """
        super().__init__()
        self.root = Path(root)
        self.audio_file_paths = []
        self.feature_paths = []
        self.format = format
        self.sizes = sizes
        self.sample_rate = sample_rate
        self.mix_down = mix_down
        self.weights_only = weights_only

        # get all paths
        for path in self.root.glob(f"**/*.{self.format}"):
            if os.path.exists(path.with_suffix(".pt")):
                self.audio_file_paths.append(path)
                self.feature_paths.append(path.with_suffix(".pt"))

    def __getitem__(self, idx):
        wf, sr = torchaudio.load(self.audio_file_paths[idx])

        # resample if different sample_rate
        if self.sample_rate is not None:
            if sr != self.sample_rate:
                wf = resample(wf, sr, self.sample_rate)

        if self.mix_down:
            wf = wf.sum(dim=0, keepdim=True)

        # load other features
        data = torch.load(self.feature_paths[idx], weights_only=True)

        # add waveform tensor to features dict.
        data["waveform"] = wf

        # adjust size
        for k in self.sizes.keys():
            v = self.sizes[k]
            if k in data:
                data[k] = adjust_size(data[k], v)

        return data

    def __len__(self):
        return len(self.audio_file_paths)

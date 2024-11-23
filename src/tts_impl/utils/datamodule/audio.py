import lightning
from tts_impl.utils.dataset import AudioDataset
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader, random_split

from typing import Union, List, Optional, Mapping, Any

class AudioDataModule(lightning.LightningDataModule):
    def __init__(self, root: Union[str | os.PathLike], seed: int=42, batch_size: int=1, lengths:  Union[List[float], list[int]]=[0.9, 0.05, 0.05], format: str="flac", sizes: Optional[Mapping[str, Any]] = {}, sample_rate: Optional[int] = None, **kwargs):
        super().__init__()
        self.root = Path(root)
        self.seed = seed
        self.batch_size = batch_size
        self.lengths = lengths
        self.sizes = sizes
        self.sample_rate = sample_rate
        self.format = format
        self.kwargs = kwargs

    def setup(self, stage: str):
        self.dataset = AudioDataset(root=self.root, format=self.format, sample_rate=self.sample_rate, sizes=self.sizes, **self.kwargs)
        g = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset, test_dataset = random_split(self.dataset, self.lengths, g)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        
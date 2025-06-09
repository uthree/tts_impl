import multiprocessing
import os
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Union

import lightning
import torch
from torch.utils.data import DataLoader, random_split
from tts_impl.utils.config import derive_config
from tts_impl.utils.dataset.vc import VcDataset


@derive_config
class VcDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        root: Union[str | os.PathLike],
        seed: int = 42,
        batch_size: int = 1,
        lengths: Union[List[float], list[int]] = [0.98, 0.01, 0.01],
        format: Literal["flac", "mp3", "wav", "ogg"] = "flac",
        sizes: Optional[Mapping[str, Any]] = {},
        sample_rate: Optional[int] = None,
        num_workers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        self.seed = seed
        self.batch_size = batch_size
        self.lengths = lengths
        self.sizes = sizes
        self.sample_rate = sample_rate
        self.format = format
        self.kwargs = kwargs

        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count() - 1
        else:
            self.num_workers = num_workers
        self.persistent_workers = os.name == "nt"

    def setup(self, stage: str):
        self.dataset = VcDataset(
            root=self.root,
            format=self.format,
            sample_rate=self.sample_rate,
            sizes=self.sizes,
            **self.kwargs,
        )
        g = torch.Generator().manual_seed(self.seed)
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, self.lengths, g
        )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=2,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

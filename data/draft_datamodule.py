"""PyTorch Lightning DataModule and minimal Dataset wrapper for TSP-style data.

This module provides:
- TSPFNDataset: flexible loader for `.pt`/`.pth` files stored per-split or a single file per split.
- TSPFNDataModule: LightningDataModule with optional deterministic splitting from an `all` dataset.

Feel free to adapt `_load` in `TSPFNDataset` to match your on-disk format (CSV, one-file-per-sample, etc.).
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Callable, Dict, Sequence, List, Union
from pathlib import Path
from tqdm import tqdm

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd


class TSPFNDataset(Dataset):
    """Minimal dataset for TSP-style tensors.

    Loading rules (defaults):
    - If `data_roots/` exists, load all `.pt`/`.pth` files inside.
    - Else if `data_roots.pt` exists, load that file. If it contains a tensor/list, treat each element as a sample.
    - Otherwise the dataset is empty.
    """

    def __init__(self, data_roots: str, subset: Path, transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.data_roots = data_roots
        self.transform = transform
        self.subset_path = subset
        self._load()

    def _load(self) -> None:
        # folder = os.path.join(self.data_roots)
        # if os.path.isdir(folder):
        #     files = sorted(os.listdir(folder))
        #     subsets = []
        #     for fname in files:
        #         if not (fname.endswith(".pt") or fname.endswith(".pth")):
        #             continue
        #         path = os.path.join(folder, fname)
        #         try:
        #             subsets.append(torch.load(path))
        #         except Exception:
        #             # skip unreadable files
        #             continue
        #     self.subsets = subsets
        #     return

        # fallback to single file
        path = os.path.join(self.subset_path)
        name_csv = os.path.basename(path)
        assert os.path.isfile(path), f"Dataset file not found: {path}"
        # Get number of lines in the file
        with open(path, "r") as f:
            total_lines = sum(1 for _ in f)

        list_df = []
        with tqdm(total=total_lines, desc=f"Loading {name_csv}") as pbar:
            for chunk in pd.read_csv(path, chunksize=1000):
                list_df.append(chunk)
                pbar.update(chunk.shape[0])

        df = pd.concat(list_df, ignore_index=True)
        # loaded_df = pd.read_csv(path, index_col=0)
        self.data_ts = list(df.values)
        self.num_classes = len(np.unique(df.iloc[:, -1]))
        return

    def __len__(self) -> int:
        return len(self.data_ts)

    def __getitem__(self, idx: int):
        sample = self.data_ts[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.num_classes


class TSPFNDataModule(pl.LightningDataModule):
    """LightningDataModule for TSP datasets.

    Parameters
    - data_roots: root directory for data
    - batch_size, num_workers, pin_memory: DataLoader args
    - transform: optional callable applied to subsets
    """

    def __init__(
        self,
        data_roots: str,
        subsets: Dict[Union[str, Subset], Union[str, Path]] = None,
        num_workers: int = 0,
        batch_size: int = 32,
        pin_memory: bool = True,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_roots = data_roots
        self.subsets = subsets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform
        self.seed = seed
        self.subset_list = [subset_path for _, subset_path in subsets.items()]

        self.current_dataset_idx = 0

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called on every process in distributed settings."""
        # TODO: fow now, train/val/test use the same subset. Later, we can modify to have different subsets for each.

        self.train_dataset = TSPFNDataset(
            data_roots=self.data_roots,
            subset=self.subset_list[self.current_dataset_idx],
            transform=self.transform,
        )
        self.val_dataset = TSPFNDataset(
            data_roots=self.data_roots,
            subset=self.subset_list[self.current_dataset_idx],
            transform=self.transform,
        )
        self.test_dataset = TSPFNDataset(
            data_roots=self.data_roots,
            subset=self.subset_list[self.current_dataset_idx],
            transform=self.transform,
        )

        return

    def switch_to_next_dataset(self):
        self.current_dataset_idx += 1
        if self.current_dataset_idx < len(self.subset_list):
            self.setup()
            return True
        return False

    def _dataloader(self, dataset: Dataset, shuffle: bool, batch_size: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return self._dataloader(self.val_dataset, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, shuffle=False, batch_size=self.batch_size)


__all__ = ["TSPFNDataset", "TSPFNDataModule"]

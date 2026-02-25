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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from data.evaluation_datasets import (
    TUABDataset,
    TUEVDataset,
    FilteredTUEVDataset,
    ECG5000Dataset,
    ESRDataset,
    ABIDEDataset,
)
from data.utils.sampler import StratifiedBatchSampler
from data.utils.processing_csv import load_csv


def stratified_batch_collate(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.tensor(ys)

    unique_labels = torch.unique(ys)
    first_half_idxs = []
    second_half_idxs = []

    for label in unique_labels:
        label_indices = (ys == label).nonzero(as_tuple=True)[0]

        # Split these specific label indices in half
        mid = len(label_indices) // 2
        first_half_idxs.append(label_indices[:mid])
        second_half_idxs.append(label_indices[mid:])

    # Combine indices for both halves
    new_order = torch.cat(first_half_idxs + second_half_idxs)

    return xs[new_order], ys[new_order]


# class TSPFNDataset(Dataset):
#     """Minimal dataset for TSP-style tensors.

#     Loading rules (defaults):
#     - If `data_roots/` exists, load all `.pt`/`.pth` files inside.
#     - Else if `data_roots.pt` exists, load that file. If it contains a tensor/list, treat each element as a sample.
#     - Otherwise the dataset is empty.
#     """

#     def __init__(
#         self, data_roots: str, subsets: List[Path], split: str, split_ratio: float, transform: Optional[Callable] = None
#     ) -> None:
#         super().__init__()
#         self.data_roots = data_roots
#         self.transform = transform
#         self.subset_paths = subsets
#         self.split = split
#         self.split_ratio = split_ratio
#         self.label_encoder = LabelEncoder()

#         data_list = []
#         for subset_path in self.subset_paths:
#             data_ts = self._load_subset(subset_path)
#             data_list.extend(data_ts)
#         self.data_ts = data_list

#     def __len__(self) -> int:
#         return len(self.data_ts)

#     def __getitem__(self, idx: int):
#         sample = self.data_ts[idx]
#         if self.transform:
#             sample = self.transform(sample)
#         # Separate features and labels
#         features = sample[:, :-1]
#         labels = sample[:, -1]
#         return features, labels


class TSPFNDataset(Dataset):
    def __init__(self, data_ts: List[torch.Tensor], transform: Optional[Callable] = None):
        super().__init__()
        self.data_ts = data_ts
        self.transform = transform

    def __len__(self):
        return len(self.data_ts)

    def __getitem__(self, idx):
        sample = self.data_ts[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample[:, :-1], sample[:, -1]


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
        test_batch_size: Optional[int] = None,
        pin_memory: bool = True,
        transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_roots = data_roots
        self.subsets = subsets
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform
        self.seed = seed
        if subsets is not None:
            self.subset_list = [subset_path for _, subset_path in subsets.items()]
        else:
            self.subset_list = [Path(data_roots)]
        self.current_dataset_idx = 0

    # def setup(self, stage: Optional[str] = None) -> None:
    #     """Create datasets. Called on every process in distributed settings."""
    #     # TODO: fow now, train/val/test use the same subset. Later, we can modify to have different subsets for each.

    #     self.train_dataset = TSPFNDataset(
    #         data_roots=self.data_roots,
    #         subsets=self.subset_list,
    #         split="train",
    #         split_ratio=0.8,
    #         transform=self.transform,
    #     )
    #     self.val_dataset = TSPFNDataset(
    #         data_roots=self.data_roots,
    #         subsets=self.subset_list,
    #         split="val",
    #         split_ratio=0.8,
    #         transform=self.transform,
    #     )
    #     self.test_dataset = TSPFNDataset(
    #         data_roots=self.data_roots,
    #         subsets=self.subset_list,
    #         split="val",
    #         split_ratio=0.8,
    #         transform=self.transform,
    #     )

    #     return

    def setup(self, stage: Optional[str] = None) -> None:
        if not hasattr(self, "all_train_chunks") or not hasattr(self, "all_val_chunks"):
            self.all_train_chunks = []
            self.all_val_chunks = []

            for subset_path in self.subset_list:
                train_chunks, val_chunks = load_csv(subset_path, split_ratio=0.8)
                self.all_train_chunks.extend(train_chunks)
                self.all_val_chunks.extend(val_chunks)

        if stage == "fit" or stage is None:
            self.train_dataset = TSPFNDataset(data_ts=self.all_train_chunks)
            self.val_dataset = TSPFNDataset(data_ts=self.all_val_chunks)

        if stage == "test":
            self.test_dataset = TSPFNDataset(data_ts=self.all_val_chunks)

    # def switch_to_next_dataset(self):
    #     self.current_dataset_idx += 1
    #     if self.current_dataset_idx < len(self.subset_list):
    #         self.setup()
    #         return True
    #     return False

    def _dataloader(
        self, dataset: Dataset, shuffle: bool, batch_size: int, collate_fn=None, drop_last=False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        val_loader = self._dataloader(self.val_dataset, shuffle=False, batch_size=self.batch_size)
        train_loader = self._dataloader(self.train_dataset, shuffle=False, batch_size=self.batch_size)
        return CombinedLoader({"val": val_loader, "train": train_loader}, "min_size")

    def test_dataloader(self):
        test_loader = self._dataloader(self.test_dataset, shuffle=False, batch_size=self.batch_size)
        train_loader = self._dataloader(self.train_dataset, shuffle=False, batch_size=self.batch_size)
        return CombinedLoader({"val": test_loader, "train": train_loader}, "min_size")


class ECG5000DataModule(TSPFNDataModule):
    """LightningDataModule for ECG5000 dataset.

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
        **kwargs,
    ) -> None:
        super().__init__(
            data_roots=data_roots,
            subsets=subsets,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=transform,
            seed=seed,
        )

        print(f"num workers: {self.num_workers}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called on every process in distributed settings."""
        self.train_dataset = ECG5000Dataset(
            root=self.data_roots,
            split="train",
        )
        self.val_dataset = ECG5000Dataset(
            root=self.data_roots,
            split="test",
        )

        return

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True, batch_size=len(self.train_dataset))

    def val_dataloader(self):
        if self.test_batch_size is None:
            self.test_batch_size = len(self.val_dataset)
        loaders = {
            "val": self._dataloader(self.val_dataset, shuffle=False, batch_size=self.test_batch_size),
            "train": self._dataloader(self.train_dataset, shuffle=False, batch_size=len(self.train_dataset)),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def test_dataloader(self):
        # This is identical to val_dataloader for the final evaluation
        return self.val_dataloader()


class ESRDataModule(TSPFNDataModule):
    """LightningDataModule for ESR dataset.

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
        **kwargs,
    ) -> None:
        super().__init__(
            data_roots=data_roots,
            subsets=subsets,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=transform,
            seed=seed,
        )

        print(f"num workers: {self.num_workers}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called on every process in distributed settings."""
        self.train_dataset = ESRDataset(
            root=self.data_roots,
            split="train",
        )
        self.val_dataset = ESRDataset(
            root=self.data_roots,
            split="test",
        )

        return

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True, batch_size=len(self.train_dataset))

    def val_dataloader(self):
        if self.test_batch_size is None:
            self.test_batch_size = len(self.val_dataset)
        loaders = {
            "val": self._dataloader(self.val_dataset, shuffle=False, batch_size=self.test_batch_size),
            "train": self._dataloader(self.train_dataset, shuffle=False, batch_size=len(self.train_dataset)),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def test_dataloader(self):
        # This is identical to val_dataloader for the final evaluation
        return self.val_dataloader()


class ABIDEDataModule(TSPFNDataModule):
    """LightningDataModule for ABIDE dataset.

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
        **kwargs,
    ) -> None:
        super().__init__(
            data_roots=data_roots,
            subsets=subsets,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=transform,
            seed=seed,
        )

        print(f"num workers: {self.num_workers}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called on every process in distributed settings."""
        self.train_dataset = ABIDEDataset(
            root=self.data_roots,
            split="train",
        )
        self.val_dataset = ABIDEDataset(
            root=self.data_roots,
            split="test",
        )

        return

    def train_dataloader(self):
        return self._dataloader(self.train_dataset, shuffle=True, batch_size=len(self.train_dataset))

    def val_dataloader(self):
        if self.test_batch_size is None:
            self.test_batch_size = len(self.val_dataset)
        loaders = {
            "val": self._dataloader(self.val_dataset, shuffle=False, batch_size=self.test_batch_size),
            "train": self._dataloader(self.train_dataset, shuffle=False, batch_size=len(self.train_dataset)),
        }
        return CombinedLoader(loaders, mode="max_size_cycle")

    def test_dataloader(self):
        # This is identical to val_dataloader for the final evaluation
        return self.val_dataloader()

class FineTuneTUEVDataModule(TSPFNDataModule):
    """LightningDataModule for TSP datasets during finetuning.

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
        super().__init__(
            data_roots=data_roots,
            subsets=subsets,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=transform,
            seed=seed,
        )

        print(f"num workers: {self.num_workers}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called on every process in distributed settings."""
        self.train_dataset = TUEVDataset(
            root=os.path.join(self.data_roots, "processed_train"),
            files=os.listdir(os.path.join(self.data_roots, "processed_train")),
            sampling_rate=200,
        )
        self.val_dataset = TUEVDataset(
            root=os.path.join(self.data_roots, "processed_eval"),
            files=os.listdir(os.path.join(self.data_roots, "processed_eval")),
            sampling_rate=200,
        )
        self.test_dataset = TUEVDataset(
            root=os.path.join(self.data_roots, "processed_test"),
            files=os.listdir(os.path.join(self.data_roots, "processed_test")),
            sampling_rate=200,
        )
        return

    def train_dataloader(self):
        return self._dataloader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=True,
        )

    def val_dataloader(self):
        val_loader = self._dataloader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=True,
        )
        train_loader = self._dataloader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=True,
        )
        return CombinedLoader({"val": val_loader, "train": train_loader}, "min_size")

    def test_dataloader(self):
        test_loader = self._dataloader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=True,
        )
        train_loader = self._dataloader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=True,
        )
        return CombinedLoader({"val": test_loader, "train": train_loader}, "min_size")


class FineTuneFilteredTUEVDataModule(TSPFNDataModule):
    """LightningDataModule for TSP datasets during finetuning.

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
        super().__init__(
            data_roots=data_roots,
            subsets=subsets,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=transform,
            seed=seed,
        )

        print(f"num workers: {self.num_workers}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called on every process in distributed settings."""
        self.train_dataset = FilteredTUEVDataset(
            root=os.path.join(self.data_roots, "processed_train"),
            files=os.listdir(os.path.join(self.data_roots, "processed_train")),
            sampling_rate=200,
        )
        self.val_dataset = FilteredTUEVDataset(
            root=os.path.join(self.data_roots, "processed_eval"),
            files=os.listdir(os.path.join(self.data_roots, "processed_eval")),
            sampling_rate=200,
        )
        self.test_dataset = FilteredTUEVDataset(
            root=os.path.join(self.data_roots, "processed_test"),
            files=os.listdir(os.path.join(self.data_roots, "processed_test")),
            sampling_rate=200,
        )
        return

    def train_dataloader(self):
        return self._dataloader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=True,
        )

    def val_dataloader(self):
        val_loader = self._dataloader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=True,
        )
        train_loader = self._dataloader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=True,
        )
        return CombinedLoader({"val": val_loader, "train": train_loader}, "min_size")

    def test_dataloader(self):
        test_loader = self._dataloader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=True,
        )
        train_loader = self._dataloader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=True,
        )
        return CombinedLoader({"val": test_loader, "train": train_loader}, "min_size")


class StratifiedFineTuneTUEVDataModule(TSPFNDataModule):
    """LightningDataModule for TSP datasets during finetuning with stratified batch sampling.

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
        super().__init__(
            data_roots=data_roots,
            subsets=subsets,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=transform,
            seed=seed,
        )

        print(f"num workers: {self.num_workers}")

    def get_stratified_sampler(self, dataset, stage: str) -> StratifiedBatchSampler:
        """Create a StratifiedBatchSampler to achieve stratified sampling."""
        labels = []
        for label in tqdm(dataset, total=len(dataset), desc=f"Extracting labels for {stage} set"):
            labels.append(label[1])
        # labels = [label for _, label in dataset]
        print(f"Class distribution in {stage} set: {pd.Series(labels).value_counts().to_dict()}")
        sampler = StratifiedBatchSampler(labels, batch_size=self.batch_size)
        return sampler

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called on every process in distributed settings."""
        self.train_dataset = TUEVDataset(
            root=os.path.join(self.data_roots, "processed_train"),
            files=os.listdir(os.path.join(self.data_roots, "processed_train")),
            sampling_rate=200,
        )
        self.val_dataset = TUEVDataset(
            root=os.path.join(self.data_roots, "processed_eval"),
            files=os.listdir(os.path.join(self.data_roots, "processed_eval")),
            sampling_rate=200,
        )
        self.test_dataset = TUEVDataset(
            root=os.path.join(self.data_roots, "processed_test"),
            files=os.listdir(os.path.join(self.data_roots, "processed_test")),
            sampling_rate=200,
        )
        self.train_sampler = self.get_stratified_sampler(self.train_dataset, stage="training")
        self.val_sampler = self.get_stratified_sampler(self.val_dataset, stage="validation")
        self.test_sampler = self.get_stratified_sampler(self.test_dataset, stage="testing")
        return

    def _train_dataloader(
        self, dataset: Dataset, batch_sampler: StratifiedBatchSampler, collate_fn=None, drop_last=False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def _eval_dataloader(self, dataset: Dataset, batch_size: int, collate_fn=None, drop_last=False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )

    def train_dataloader(self):
        return self._train_dataloader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=None,
            drop_last=False,
        )

    def val_dataloader(self):
        val_loader = self._eval_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=False,
        )
        train_loader = self._train_dataloader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=None,
            drop_last=False,
        )
        return CombinedLoader({"val": val_loader, "train": train_loader}, "min_size")

    def test_dataloader(self):
        test_loader = self._eval_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=None,
            drop_last=False,
        )
        train_loader = self._train_dataloader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=None,
            drop_last=False,
        )
        return CombinedLoader({"val": test_loader, "train": train_loader}, "min_size")


class WeightedFineTuneTUEVDataModule(TSPFNDataModule):
    """LightningDataModule for TSP datasets during finetuning.

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
        super().__init__(
            data_roots=data_roots,
            subsets=subsets,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            transform=transform,
            seed=seed,
        )

        print(f"num workers: {self.num_workers}")

    def get_stratified_sampler(self, dataset, stage: str) -> WeightedRandomSampler:
        """Create a WeightedRandomSampler to achieve stratified sampling."""
        labels = [label for _, label in dataset]
        labels = torch.tensor(labels)
        class_counts = torch.bincount(labels)
        print(f"Class distribution in {stage} set: {class_counts.tolist()}")
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return sampler

    def setup(self, stage: Optional[str] = None) -> None:
        """Create datasets. Called on every process in distributed settings."""
        self.train_dataset = TUEVDataset(
            root=os.path.join(self.data_roots, "processed_train"),
            files=os.listdir(os.path.join(self.data_roots, "processed_train")),
            sampling_rate=200,
        )
        self.val_dataset = TUEVDataset(
            root=os.path.join(self.data_roots, "processed_eval"),
            files=os.listdir(os.path.join(self.data_roots, "processed_eval")),
            sampling_rate=200,
        )
        self.test_dataset = TUEVDataset(
            root=os.path.join(self.data_roots, "processed_test"),
            files=os.listdir(os.path.join(self.data_roots, "processed_test")),
            sampling_rate=200,
        )
        self.train_sampler = self.get_stratified_sampler(self.train_dataset, stage="training")
        self.val_sampler = self.get_stratified_sampler(self.val_dataset, stage="validation")
        self.test_sampler = self.get_stratified_sampler(self.test_dataset, stage="testing")
        return

    def _dataloader(
        self, dataset: Dataset, batch_size: int, sampler: WeightedRandomSampler, collate_fn=None, drop_last=False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
            shuffle=False,  # Use weighted sampler instead
        )

    def train_dataloader(self):
        return self._dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            collate_fn=None,
            drop_last=True,
        )

    def val_dataloader(self):
        val_loader = self._dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            collate_fn=None,
            drop_last=True,
        )
        train_loader = self._dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            collate_fn=None,
            drop_last=True,
        )
        return CombinedLoader({"val": val_loader, "train": train_loader}, "min_size")

    def test_dataloader(self):
        test_loader = self._dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            collate_fn=None,
            drop_last=True,
        )
        train_loader = self._dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            collate_fn=None,
            drop_last=True,
        )
        return CombinedLoader({"val": test_loader, "train": train_loader}, "min_size")


__all__ = ["TSPFNDataset", "TSPFNDataModule", "FineTuneDataModule"]

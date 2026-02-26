# --------------------------------------------------------
# Based on LaBraM, BEiT-v2, timm, DeiT, DINO, and BIOT code bases
# https://github.com/935963004/LaBraM
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/ycq091044/BIOT
# ---------------------------------------------------------

import io
import os
import math
import time
import json
from glob import glob
from collections import defaultdict, deque
import datetime
import numpy as np
import pandas as pd
from scipy.io import arff
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import argparse

import torch
import torch.distributed as dist
from torch import inf

import pickle
import scipy.signal as sgn
from scipy.signal import resample
from torch.utils.data import Dataset

standard_1020 = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]


class TUABDataset(Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class TUEVDataset(Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        # Normalize by 100
        # X = X / 100.0
        Y = int(sample["label"][0] - 1)  # make label start from 0
        X = torch.FloatTensor(X)
        return X, Y


class FilteredTUEVDataset(Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 2 * self.sampling_rate, axis=-1)
        Y = int(sample["label"] - 1)  # make label start from 0
        X = torch.FloatTensor(X)
        mask = sample["mask"]
        mask = torch.FloatTensor(mask)
        return X, Y, mask


class ECG5000Dataset(Dataset):
    def __init__(
        self, root, split: str, scaler=None, support_size=None, fold=None, present_labels=None
    ):  # Added scaler argument
        self.root = root
        self.file_path = os.path.join(self.root, f"{split}", f"{split}.csv")

        df = pd.read_csv(self.file_path, index_col=0)
        self.data = df.values
        self.present_labels = present_labels
        print(f"Count labels in {split} split before subsampling: {np.unique(self.data[:, -1], return_counts=True)}")

        if support_size is not None and split == "train":
            unique_labels = np.unique(self.data[:, -1])
            n_folds = 5
            min_per_class = 2  # Ensuring fold safety

            # Create a deterministic shuffled order for every class
            # We use a fixed seed so the "order" is the same every time you run this
            rng = np.random.default_rng(42)

            # This dictionary will store the indices for each class, pre-shuffled
            class_indices = {}
            for label in unique_labels:
                idx = np.where(self.data[:, -1] == label)[0]
                rng.shuffle(idx)
                class_indices[label] = idx

            selected_indices = []

            # Mandatory "Safety" Pick (Small classes first)
            # This ensures Class 5 always gets its 10-19 samples regardless of total size
            for label in unique_labels:
                n_to_take = min(len(class_indices[label]), min_per_class)
                selected_indices.extend(class_indices[label][:n_to_take])
                # Remove these from the available pool
                class_indices[label] = class_indices[label][n_to_take:]

            # Global "Greedy" Fill
            # Combine everything else left into one big pool and shuffle it once
            remaining_pool = np.concatenate(list(class_indices.values()))
            rng.shuffle(remaining_pool)

            # Calculate how many more we need to hit the target support_size
            needed = support_size - len(selected_indices)

            if needed > 0:
                # Take the top 'N' from the remaining pool
                selected_indices.extend(remaining_pool[:needed])

            self.data = self.data[selected_indices]
            # Optional: shuffle the final data so the model doesn't see classes in order
            rng.shuffle(self.data)

        self.X = self.data[:, :-1]
        self.Y = self.data[:, -1].astype(int) - 1
        if self.present_labels is not None:
            filtered_indices = []
            filtered_labels = []
            for idx, label in enumerate(self.Y):
                if label in self.present_labels:
                    filtered_indices.append(idx)
                    filtered_labels.append(label)
            print(f"Filtering to present labels {self.present_labels}: {len(filtered_indices)} samples remain.")
            self.X = self.X[filtered_indices]
            self.Y = np.array(filtered_labels)
            # Use LabelEncoder to enforce label to range from 0 to num_classes-1 after filtering
            le = LabelEncoder()
            self.Y = le.fit_transform(self.Y)

        if fold is not None and split == "train":
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            list_of_split = list(skf.split(self.X, self.Y))
            self.X = self.X[list_of_split[fold][1]]  # Use the specified fold's test indices for validation
            self.Y = self.Y[list_of_split[fold][1]]
            # Use LabelEncoder to enforce label to range from 0 to num_classes-1 after filtering
            le = LabelEncoder()
            self.Y = le.fit_transform(self.Y)
            self.present_labels = np.unique(self.Y)

        print(f"Count labels in {split} split: {np.unique(self.Y, return_counts=True)}")

        if self.X.ndim == 2:
            self.X = self.X.reshape(
                self.X.shape[0], 1, -1
            )  # Add unichannel dimension if missing, shape becomes (N, 1, Time)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x_sample = self.X[index]
        y_sample = self.Y[index]

        x_tensor = torch.as_tensor(x_sample, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_sample, dtype=torch.long)

        return x_tensor, y_tensor


class ESRDataset(Dataset):
    def __init__(self, root, split: str, scaler=None, support_size=None, fold=None):  # Added scaler argument
        self.root = root
        self.file_path = os.path.join(self.root, f"{split}", f"{split}.csv")

        df = pd.read_csv(self.file_path, index_col=0)
        self.data = df.values

        if support_size is not None and split == "train":
            unique_labels = np.unique(self.data[:, -1])
            n_folds = 5
            min_per_class = 2  # Ensuring fold safety

            # Create a deterministic shuffled order for every class
            # We use a fixed seed so the "order" is the same every time you run this
            rng = np.random.default_rng(42)

            # This dictionary will store the indices for each class, pre-shuffled
            class_indices = {}
            for label in unique_labels:
                idx = np.where(self.data[:, -1] == label)[0]
                rng.shuffle(idx)
                class_indices[label] = idx

            selected_indices = []

            # Mandatory "Safety" Pick (Small classes first)
            # This ensures Class 5 always gets its 10-19 samples regardless of total size
            for label in unique_labels:
                n_to_take = min(len(class_indices[label]), min_per_class)
                selected_indices.extend(class_indices[label][:n_to_take])
                # Remove these from the available pool
                class_indices[label] = class_indices[label][n_to_take:]

            # Global "Greedy" Fill
            # Combine everything else left into one big pool and shuffle it once
            remaining_pool = np.concatenate(list(class_indices.values()))
            rng.shuffle(remaining_pool)

            # Calculate how many more we need to hit the target support_size
            needed = support_size - len(selected_indices)

            if needed > 0:
                # Take the top 'N' from the remaining pool
                selected_indices.extend(remaining_pool[:needed])

            # Apply
            self.data = self.data[selected_indices]
            # Optional: shuffle the final data so the model doesn't see classes in order
            rng.shuffle(self.data)

        self.X = self.data[:, :-1]
        self.Y = self.data[:, -1].astype(int) - 1

        if fold is not None and split == "train":
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            list_of_split = list(skf.split(self.X, self.Y))
            self.X = self.X[list_of_split[fold][1]]  # Use the specified fold's test indices for validation
            self.Y = self.Y[list_of_split[fold][1]]

        print(f"Count labels in {split} split: {np.unique(self.Y, return_counts=True)}")

        if self.X.ndim == 2:
            self.X = self.X.reshape(
                self.X.shape[0], 1, -1
            )  # Add unichannel dimension if missing, shape becomes (N, 1, Time)

        self.scaler = None

        print(f"Loaded {len(self.X)} samples for {split} split of ESR dataset.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x_sample = self.X[index]
        y_sample = self.Y[index]

        x_tensor = torch.as_tensor(x_sample, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_sample, dtype=torch.long)

        return x_tensor, y_tensor


class EICUCRDDataset(Dataset):
    def __init__(self, root, split: str, support_size=None, fold=None):
        self.root = root
        self.support_size = support_size
        self.file_dir = os.path.join(self.root, f"{split}_decease")
        self.label_file = os.path.join(self.root, "final_labels.csv")
        self.selected_channels = ["heart_rate", "respiration", "spo2", "blood_pressure", "temperature"]

        channel_maps = {
            "heart_rate": 0,
            "respiration": 1,
            "spo2": 2,
            "blood_pressure": 3,
            "temperature": 4,
        }

        self.all_patients = sorted(glob(os.path.join(self.file_dir, "*.npz")))
        print(f"Found {len(self.all_patients)} files in {self.file_dir}")
        patient_dict = {}
        for patient in self.all_patients:
            data = np.load(patient)["data"]
            if len(self.selected_channels) < 5:
                channel_idxs = [channel_maps[ch] for ch in self.selected_channels]
                data = data[:, channel_idxs]
            patient_dict[Path(patient).stem] = data.T  # Transpose to get shape (Channels, Time)
        self.patient_dict = patient_dict
        print(f"data shape after loading: {data.T.shape}")
        self.df_labels = pd.read_csv(self.label_file, index_col=0)

        if support_size is not None and split == "train":
            unique_labels = np.unique(
                self.df_labels.loc[[int(Path(p).stem) for p in self.all_patients], "mortality_label"]
            )
            n_folds = 5
            min_per_class = 2  # Ensuring fold safety

            rng = np.random.default_rng(42)

            class_indices = {}
            for label in unique_labels:
                idx = np.where(
                    self.df_labels.loc[[int(Path(p).stem) for p in self.all_patients], "mortality_label"] == label
                )[0]
                rng.shuffle(idx)
                class_indices[label] = idx

            selected_indices = []
            for label in unique_labels:
                n_to_take = min(len(class_indices[label]), min_per_class)
                selected_indices.extend(class_indices[label][:n_to_take])
                class_indices[label] = class_indices[label][n_to_take:]

            remaining_pool = np.concatenate(list(class_indices.values()))
            rng.shuffle(remaining_pool)

            needed = support_size - len(selected_indices)
            if needed > 0:
                selected_indices.extend(remaining_pool[:needed])

            # rng.shuffle(self.all_patients)  # Shuffle the order of patients after subsampling
            self.all_patients = [self.all_patients[i] for i in selected_indices]
            print(f"Subsampling {len(selected_indices)} samples from {len(self.all_patients)} for training.")
            print(
                f"Count labels in subsampled training set: {np.unique(self.df_labels.loc[[int(Path(p).stem) for p in self.all_patients], 'mortality_label'], return_counts=True)}"
            )

        if fold is not None and split == "train":
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            list_of_split = list(
                skf.split(
                    self.all_patients,
                    self.df_labels.loc[[int(Path(p).stem) for p in self.all_patients], "mortality_label"],
                )
            )
            self.all_patients = [
                self.all_patients[i] for i in list_of_split[fold][1]
            ]  # Use the specified fold's test indices for validation
            self.patient_dict = {Path(p).stem: self.patient_dict[Path(p).stem] for p in self.all_patients}
            self.df_labels = self.df_labels.loc[[int(Path(p).stem) for p in self.all_patients]]

    def __len__(self):
        return len(self.all_patients)

    def __getitem__(self, index):
        file_path = self.all_patients[index]
        file_name = Path(file_path).stem
        x_sample = self.patient_dict[file_name]
        y_sample = self.df_labels.loc[int(file_name), "mortality_label"]  # Labels are indexed by file name

        x_tensor = torch.as_tensor(x_sample, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_sample, dtype=torch.long)

        return x_tensor, y_tensor


class EOSDataset(Dataset):
    def __init__(self, root, split: str, support_size=None, fold=None):
        self.root = root

        self.X = np.load(os.path.join(self.root, f"{split}_features.npy"))
        self.Y = np.load(os.path.join(self.root, f"{split}_labels.npy")).astype(int)

        if support_size is not None and split == "train":
            X_train = np.load(os.path.join(self.root, f"train_features.npy"))
            Y_train = np.load(os.path.join(self.root, f"train_labels.npy")).astype(int)
            X_test = np.load(os.path.join(self.root, f"test_features.npy"))
            Y_test = np.load(os.path.join(self.root, f"test_labels.npy")).astype(int)
            X_full = np.concatenate([X_train, X_test], axis=0)
            Y_full = np.concatenate([Y_train, Y_test], axis=0)

            unique_labels = np.unique(Y_full)
            n_folds = 5
            min_per_class = 2  # Ensuring fold safety

            # Create a deterministic shuffled order for every class
            # We use a fixed seed so the "order" is the same every time you run this
            rng = np.random.default_rng(42)

            # This dictionary will store the indices for each class, pre-shuffled
            class_indices = {}
            for label in unique_labels:
                idx = np.where(Y_full == label)[0]
                rng.shuffle(idx)
                class_indices[label] = idx

            selected_indices = []

            # Mandatory "Safety" Pick (Small classes first)
            # This ensures Class 5 always gets its 10-19 samples regardless of total size
            for label in unique_labels:
                n_to_take = min(len(class_indices[label]), min_per_class)
                selected_indices.extend(class_indices[label][:n_to_take])
                # Remove these from the available pool
                class_indices[label] = class_indices[label][n_to_take:]

            # Global "Greedy" Fill
            # Combine everything else left into one big pool and shuffle it once
            remaining_pool = np.concatenate(list(class_indices.values()))
            rng.shuffle(remaining_pool)

            # Calculate how many more we need to hit the target support_size
            needed = support_size - len(selected_indices)

            if needed > 0:
                # Take the top 'N' from the remaining pool
                selected_indices.extend(remaining_pool[:needed])

            self.X = X_full[selected_indices]
            self.Y = Y_full[selected_indices]
            # Optional: shuffle the final data so the model doesn't see classes in order
            indices = np.arange(len(self.X))
            rng.shuffle(indices)
            self.X = self.X[indices]
            self.Y = self.Y[indices]

        if fold is not None and split == "train":
            assert support_size is not None
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            list_of_split = list(skf.split(self.X, self.Y))
            self.X = self.X[list_of_split[fold][1]]  # Use the specified fold's test indices for validation
            self.Y = self.Y[list_of_split[fold][1]]
            print(f"Count labels in {split} split after fold selection: {np.unique(self.Y, return_counts=True)}")

        if self.X.shape[1] > 5:
            keep_channels = [4, 5, 11]
            print(f"-----KEEP CHANNELS: {keep_channels}")
            self.X = self.X[:, keep_channels, :]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x_sample = self.X[index]
        y_sample = self.Y[index]

        x_tensor = torch.as_tensor(x_sample, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_sample, dtype=torch.long)

        return x_tensor, y_tensor


class CPSCDataset(Dataset):
    def __init__(self, root, split, support_size=None, fold=None):
        self.root = root

        self.X = np.load(os.path.join(root, f"{split}.npy"))
        self.Y = np.load(os.path.join(root, f"{split}_label.npy"))
        self.X = torch.from_numpy(self.X).float()
        self.X = self.X.reshape(self.X.shape[0], 4, -1)  # Reshape to [Batch, Channels, Signal_Length]
        self.Y = torch.from_numpy(self.Y).long()  # Shape [Batch, 1]

        if support_size is not None and split == "train":
            unique_labels = np.unique(self.Y)
            n_folds = 5
            min_per_class = 2  # Ensuring fold safety

            rng = np.random.default_rng(42)

            class_indices = {}
            for label in unique_labels:
                idx = np.where(self.Y == label)[0]
                rng.shuffle(idx)
                class_indices[label] = idx

            selected_indices = []
            for label in unique_labels:
                n_to_take = min(len(class_indices[label]), min_per_class)
                selected_indices.extend(class_indices[label][:n_to_take])
                class_indices[label] = class_indices[label][n_to_take:]

            remaining_pool = np.concatenate(list(class_indices.values()))
            rng.shuffle(remaining_pool)

            needed = support_size - len(selected_indices)
            if needed > 0:
                selected_indices.extend(remaining_pool[:needed])

            self.X = self.X[selected_indices]
            self.Y = self.Y[selected_indices]
            print(f"Subsampling {len(selected_indices)} samples from {len(self.X)} for training.")
            print(f"Count labels in subsampled training set: {np.unique(self.Y, return_counts=True)}")

        if fold is not None and split == "train":
            assert support_size is not None
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            list_of_split = list(skf.split(self.X, self.Y))
            self.X = self.X[list_of_split[fold][1]]  # Use the specified fold's test indices for validation
            self.Y = self.Y[list_of_split[fold][1]]
            print(f"Count labels in {split} split after fold selection: {np.unique(self.Y, return_counts=True)}")

        print(f"Loaded CPSC dataset with {len(self.X)} samples")
        if self.X.shape[2] < 125:
            self.X = F.pad(self.X, (0, 125 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 125]
        elif self.X.shape[2] == 125:
            pass
        else:
            raise ValueError(
                f"Expected signal length of 125, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

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
    def __init__(self, root, split: str, scaler=None, support_size=None, fold=None, present_labels=None):  # Added scaler argument
        self.root = root
        self.file_path = os.path.join(self.root, f"{split}", f"{split}.csv")

        df = pd.read_csv(self.file_path, index_col=0)
        self.data = df.values
        self.present_labels = present_labels
        print(f"Count labels in {split} split before subsampling: {np.unique(self.data[:, -1], return_counts=True)}")
        # if support_size is not None and split == "train":
        #     indices = list(range(len(self.data)))
        #     train_labels = self.data[:, -1]
        #     _, sub_indices = train_test_split(
        #         indices, test_size=support_size, random_state=42, stratify=train_labels
        #     )
        #     print(f"Subsampling {support_size} samples from {len(self.data)} for training.")
        #     print(f"Chosen indices: {sub_indices[:10]}...")  # Print first 10 indices for verification
        #     self.data = self.data[sub_indices]

        if support_size is not None and split == "train":
            unique_labels = np.unique(self.data[:, -1])
            n_folds = 5
            min_per_class = 2  # Ensuring fold safety

            # 1. Create a deterministic shuffled order for every class
            # We use a fixed seed so the "order" is the same every time you run this
            rng = np.random.default_rng(42)

            # This dictionary will store the indices for each class, pre-shuffled
            class_indices = {}
            for label in unique_labels:
                idx = np.where(self.data[:, -1] == label)[0]
                rng.shuffle(idx)
                class_indices[label] = idx

            selected_indices = []

            # 2. Mandatory "Safety" Pick (Small classes first)
            # This ensures Class 5 always gets its 10-19 samples regardless of total size
            for label in unique_labels:
                n_to_take = min(len(class_indices[label]), min_per_class)
                selected_indices.extend(class_indices[label][:n_to_take])
                # Remove these from the available pool
                class_indices[label] = class_indices[label][n_to_take:]

            # 3. Global "Greedy" Fill
            # Combine everything else left into one big pool and shuffle it once
            remaining_pool = np.concatenate(list(class_indices.values()))
            rng.shuffle(remaining_pool)

            # Calculate how many more we need to hit the target support_size
            needed = support_size - len(selected_indices)

            if needed > 0:
                # Take the top 'N' from the remaining pool
                selected_indices.extend(remaining_pool[:needed])

            # 4. Apply
            self.data = self.data[selected_indices]
            # Optional: shuffle the final data so the model doesn't see classes in order
            rng.shuffle(self.data)

            # print(f"Subsampling {len(sub_indices)} samples from {len(self.data)} for training.")
            # self.data = self.data[sub_indices]

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

        # if support_size is not None and split == "train":
        #     indices = list(range(len(self.data)))
        #     _, sub_indices = train_test_split(
        #         indices, test_size=support_size, random_state=42, stratify=self.data[:, -1]
        #     )
        #     print(f"Subsampling {support_size} samples from {len(self.data)} for training.")
        #     print(f"Chosen indices: {sub_indices[:10]}...")  # Print first 10 indices for verification
        #     self.data = self.data[sub_indices]

        if support_size is not None and split == "train":
            unique_labels = np.unique(self.data[:, -1])
            n_folds = 5
            min_per_class = 2  # Ensuring fold safety

            # 1. Create a deterministic shuffled order for every class
            # We use a fixed seed so the "order" is the same every time you run this
            rng = np.random.default_rng(42)

            # This dictionary will store the indices for each class, pre-shuffled
            class_indices = {}
            for label in unique_labels:
                idx = np.where(self.data[:, -1] == label)[0]
                rng.shuffle(idx)
                class_indices[label] = idx

            selected_indices = []

            # 2. Mandatory "Safety" Pick (Small classes first)
            # This ensures Class 5 always gets its 10-19 samples regardless of total size
            for label in unique_labels:
                n_to_take = min(len(class_indices[label]), min_per_class)
                selected_indices.extend(class_indices[label][:n_to_take])
                # Remove these from the available pool
                class_indices[label] = class_indices[label][n_to_take:]

            # 3. Global "Greedy" Fill
            # Combine everything else left into one big pool and shuffle it once
            remaining_pool = np.concatenate(list(class_indices.values()))
            rng.shuffle(remaining_pool)

            # Calculate how many more we need to hit the target support_size
            needed = support_size - len(selected_indices)

            if needed > 0:
                # Take the top 'N' from the remaining pool
                selected_indices.extend(remaining_pool[:needed])

            # 4. Apply
            self.data = self.data[selected_indices]
            # Optional: shuffle the final data so the model doesn't see classes in order
            rng.shuffle(self.data)

            # print(f"Subsampling {len(sub_indices)} samples from {len(self.data)} for training.")
            # self.data = self.data[sub_indices]

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

        # if split == "train":
        #     self.scaler = StandardScaler()
        #     self.X = self.scaler.fit_transform(self.X)
        # else:
        #     # Use the passed scaler, or handle the case where it's missing
        #     if scaler is None:
        #         raise ValueError("A fitted scaler must be provided for the test/val split!")
        #     self.scaler = scaler
        #     self.X = self.scaler.transform(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x_sample = self.X[index]
        y_sample = self.Y[index]

        x_tensor = torch.as_tensor(x_sample, dtype=torch.float32)
        y_tensor = torch.as_tensor(y_sample, dtype=torch.long)

        return x_tensor, y_tensor


class ORCHIDDataset(Dataset):
    def __init__(self, root, split: str):
        self.root = root
        self.file_dir = os.path.join(self.root, f"data/{split}")
        self.label_file = os.path.join(self.root, "labels.csv")
        self.selected_channels = [
            "gls",
            "ls_left",
            # "ls_right",
            "lv_area",
            # "lv_length",
            "myo_thickness_left",
            "myo_thickness_right",
        ]

        self.all_patients = sorted(glob(os.path.join(self.file_dir, "*.npz")))
        print(f"Found {len(self.all_patients)} files in {self.file_dir}")
        patient_dict = {}
        for patient in self.all_patients:
            data = np.load(patient)
            patient_stacked = np.stack([data[channel] for channel in self.selected_channels], axis=0)
            patient_dict[Path(patient).stem] = patient_stacked
        self.patient_dict = patient_dict
        self.df_labels = pd.read_csv(self.label_file, index_col=0)

    def __len__(self):
        return len(self.all_patients)

    def __getitem__(self, index):
        file_path = self.all_patients[index]
        file_name = Path(file_path).stem
        x_sample = self.patient_dict[file_name]
        y_sample = self.df_labels.loc[int(file_name[:4]), "diagnosis"]  # Labels are indexed by file name

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

        # if support_size is not None and split == "train":
        #     indices = list(range(len(self.all_patients)))
        #     train_labels = self.df_labels.loc[[int(Path(p).stem) for p in self.all_patients], "mortality_label"]
        #     print(f"train labels shape: {train_labels.shape}")
        #     print(f"len indices: {len(indices)}")
        #     _, sub_indices = train_test_split(
        #         indices, test_size=support_size, random_state=42, stratify=train_labels
        #     )
        #     print(f"Subsampling {support_size} samples from {len(self.all_patients)} for training.")
        #     print(f"Chosen indices: {sub_indices[:10]}...")  # Print first 10 indices for verification
        #     self.all_patients = [self.all_patients[i] for i in sub_indices]
        #     print(f"Count labels in subsampled training set: {np.unique(train_labels.iloc[sub_indices], return_counts=True)}")

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

        # self.X = self.data[:, :-1]
        # self.Y = self.data[:, -1].astype(int) - 1

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


class BlinkDataset(Dataset):
    def __init__(self, root, split: str):
        self.root = root

        self.X = np.load(os.path.join(self.root, f"{split}_features.npy"))
        self.Y = np.load(os.path.join(self.root, f"{split}_labels.npy")).astype(int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x_sample = self.X[index]
        y_sample = self.Y[index]

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

            # self.data = np.column_stack([X_full, Y_full])
            unique_labels = np.unique(Y_full)
            n_folds = 5
            min_per_class = 2  # Ensuring fold safety

            # 1. Create a deterministic shuffled order for every class
            # We use a fixed seed so the "order" is the same every time you run this
            rng = np.random.default_rng(42)

            # This dictionary will store the indices for each class, pre-shuffled
            class_indices = {}
            for label in unique_labels:
                idx = np.where(Y_full == label)[0]
                rng.shuffle(idx)
                class_indices[label] = idx

            selected_indices = []

            # 2. Mandatory "Safety" Pick (Small classes first)
            # This ensures Class 5 always gets its 10-19 samples regardless of total size
            for label in unique_labels:
                n_to_take = min(len(class_indices[label]), min_per_class)
                selected_indices.extend(class_indices[label][:n_to_take])
                # Remove these from the available pool
                class_indices[label] = class_indices[label][n_to_take:]

            # 3. Global "Greedy" Fill
            # Combine everything else left into one big pool and shuffle it once
            remaining_pool = np.concatenate(list(class_indices.values()))
            rng.shuffle(remaining_pool)

            # Calculate how many more we need to hit the target support_size
            needed = support_size - len(selected_indices)

            if needed > 0:
                # Take the top 'N' from the remaining pool
                selected_indices.extend(remaining_pool[:needed])

            # 4. Apply
            self.X = X_full[selected_indices]
            self.Y = Y_full[selected_indices]
            # Optional: shuffle the final data so the model doesn't see classes in order
            indices = np.arange(len(self.X))
            rng.shuffle(indices)
            self.X = self.X[indices]
            self.Y = self.Y[indices]

            # print(f"Subsampling {len(sub_indices)} samples from {len(self.data)} for training.")
            # self.data = self.data[sub_indices]

        if fold is not None and split == "train":
            assert support_size is not None
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            list_of_split = list(skf.split(self.X, self.Y))
            self.X = self.X[list_of_split[fold][1]]  # Use the specified fold's test indices for validation
            self.Y = self.Y[list_of_split[fold][1]]
            print(f"Count labels in {split} split after fold selection: {np.unique(self.Y, return_counts=True)}")

        if self.X.shape[1] > 5:
            # Use the first 5 channels if there are more than 5
            all_channels = list(range(self.X.shape[1]))
            # keep_channels = [0, 10, 11, 12, 13]
            # keep_channels = [0, 10, 11, 12]
            # keep_channels = [10, 11, 12]
            # keep_channels = [7, 4, 1]
            # Good below
            # keep_channels = [0, 10, 11]
            # keep_channels = [4, 6, 8]
            # keep_chahnels = [3, 6, 7]
            # VERY GOOD BELOW
            # keep_channels = [11, 4, 5]
            # np.random.seed(11)  # Set seed for reproducibility
            # keep_channels = np.random.choice(all_channels, size=3, replace=False)
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


class AtrialFibrillationDataset(Dataset):
    def _preprocess_ecg(self, x, fs=128):  # fs = sampling frequency
        # 1. High-pass filter (0.5 Hz)
        b, a = sgn.butter(3, 0.5 / (fs / 2), "high")
        x = sgn.filtfilt(b, a, x)

        # 2. Notch filter (50Hz or 60Hz)
        b_notch, a_notch = sgn.iirnotch(50 / (fs / 2), 30)
        x = sgn.filtfilt(b_notch, a_notch, x)
        x = sgn.decimate(x, q=2, axis=-1)
        x = resample(x, num=250, axis=-1)  # Resample to 250 time points
        # Truncate to 250 time points

        return x

    # def _preprocess_ecg(self, x, fs=128):
    #     # 1. High-pass (0.5 Hz) -> Remove breathing drift
    #     b, a = sgn.butter(3, 0.5 / (fs / 2), "high")
    #     x = sgn.filtfilt(b, a, x)

    #     # 2. Low-pass (40 Hz) -> CRITICAL: Remove muscle noise/EMG
    #     # AFib signals are very noisy; this cleans the "fuzz"
    #     b_lp, a_lp = sgn.butter(3, 40.0 / (fs / 2), "low")
    #     x = sgn.filtfilt(b_lp, a_lp, x)

    #     # 3. Notch filter (50Hz) -> Remove power line hum
    #     b_n, a_n = sgn.iirnotch(50 / (fs / 2), 30)
    #     x = sgn.filtfilt(b_n, a_n, x)

    #     # 4. Polyphase Resampling (Cleaner than FFT resample)
    #     # Target 250, Current 640 -> Ratio is 25/64
    #     x = sgn.resample_poly(x, 25, 64, axis=-1)

    #     return x

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

            # self.data = np.column_stack([X_full, Y_full])
            unique_labels = np.unique(Y_full)
            n_folds = 5
            min_per_class = 2  # Ensuring fold safety

            # 1. Create a deterministic shuffled order for every class
            # We use a fixed seed so the "order" is the same every time you run this
            rng = np.random.default_rng(42)

            # This dictionary will store the indices for each class, pre-shuffled
            class_indices = {}
            for label in unique_labels:
                idx = np.where(Y_full == label)[0]
                rng.shuffle(idx)
                class_indices[label] = idx

            selected_indices = []

            # 2. Mandatory "Safety" Pick (Small classes first)
            # This ensures Class 5 always gets its 10-19 samples regardless of total size
            for label in unique_labels:
                n_to_take = min(len(class_indices[label]), min_per_class)
                selected_indices.extend(class_indices[label][:n_to_take])
                # Remove these from the available pool
                class_indices[label] = class_indices[label][n_to_take:]

            # 3. Global "Greedy" Fill
            # Combine everything else left into one big pool and shuffle it once
            remaining_pool = np.concatenate(list(class_indices.values()))
            rng.shuffle(remaining_pool)

            # Calculate how many more we need to hit the target support_size
            needed = support_size - len(selected_indices)

            if needed > 0:
                # Take the top 'N' from the remaining pool
                selected_indices.extend(remaining_pool[:needed])

            # 4. Apply
            self.X = X_full[selected_indices]
            self.Y = Y_full[selected_indices]
            # Optional: shuffle the final data so the model doesn't see classes in order
            indices = np.arange(len(self.X))
            rng.shuffle(indices)
            self.X = self.X[indices]
            self.Y = self.Y[indices]

            # print(f"Subsampling {len(sub_indices)} samples from {len(self.data)} for training.")
            # self.data = self.data[sub_indices]

        if fold is not None and split == "train":
            assert support_size is not None
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            list_of_split = list(skf.split(self.X, self.Y))
            self.X = self.X[list_of_split[fold][1]]  # Use the specified fold's test indices for validation
            self.Y = self.Y[list_of_split[fold][1]]
            print(f"Count labels in {self.split} split after fold selection: {np.unique(self.Y, return_counts=True)}")

        self.X = np.array([self._preprocess_ecg(x) for x in self.X])

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


# class ABIDEDataset(Dataset):
#     def __init__(self, root, split: str):
#         self.root = root
#         self.file_path_directory = os.path.join(self.root, f"{split}_pca")
#         self.label_file = os.path.join(self.root, "labels.csv")

#         self.all_files = glob(os.path.join(self.file_path_directory, "*.npy"))
#         print(f"Found {len(self.all_files)} files in {self.file_path_directory}")
#         self.df_labels = pd.read_csv(self.label_file, index_col=0)

#     def __len__(self):
#         return len(self.all_files)

#     def __getitem__(self, index):
#         file_path = self.all_files[index]
#         x_sample = np.load(file_path)
#         file_name = Path(file_path).stem
#         y_sample = self.df_labels.loc[int(file_name), "target"]  # Labels are already zero-based indexed

#         x_tensor = torch.as_tensor(x_sample, dtype=torch.float32)
#         y_tensor = torch.as_tensor(y_sample, dtype=torch.long)

#         return x_tensor, y_tensor


class ABIDEDataset(Dataset):
    def __init__(self, root, split: str):
        self.root = root
        self.split = split
        self.file_dir = os.path.join(self.root, f"{split}_pca")
        self.label_file = os.path.join(self.root, "labels.csv")

        self.all_files = sorted(glob(os.path.join(self.file_dir, "*.npy")))
        self.df_labels = pd.read_csv(self.label_file, index_col=0)

        # Lists to hold the actual data
        self.data = []
        self.labels = []
        self.ids = []

        self._load_data()

    def _load_data(self):
        print(f"--- Loading {self.split} set into RAM ---")
        # tqdm gives you the progress bar you requested
        for f_path in tqdm(self.all_files, desc=f"Loading {self.split}"):
            file_name = Path(f_path).stem
            sub_id = int(file_name)

            # Load and convert
            x_sample = np.load(f_path)
            y_sample = self.df_labels.loc[sub_id, "target"]

            self.data.append(x_sample)
            self.labels.append(y_sample)
            self.ids.append(sub_id)

        # Convert list of arrays into a single float32 tensor
        # Shape: (N, Time, PCA_Components)
        # self.data = torch.from_numpy(np.array(self.data)[:, [0, 1], :]).float()
        self.data = torch.from_numpy(np.array(self.data)).float()
        self.data = self.data.reshape(self.data.size(0), 5, -1)  # Flatten to (N, PCA_Components, Time)
        print(f"Data shape after loading: {self.data.shape}")
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # O(1) complexity since data is already in memory as a tensor
        return self.data[index], self.labels[index]

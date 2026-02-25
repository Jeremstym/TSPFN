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
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Union, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import inf

import pickle
from scipy.signal import resample
from torch.utils.data import Dataset, DataLoader


class PTB2ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        # self.files = glob(os.path.join(root, f"{split}/*.pkl"))

        self.X = np.load(os.path.join(root, f"{split}.npy"))
        # Subsample time series
        indices = range(len(self.X))
        sub_indices = np.random.choice(indices, size=int(len(self.X) * 0.3), replace=False)
        self.X = self.X[sub_indices]
        self.Y = np.load(os.path.join(root, f"{split}_label.npy"))
        # self.Y = self.Y[sub_indices]
        self.X = torch.from_numpy(self.X).float()
        self.X = self.X.reshape(self.X.shape[0], 2, -1)  # Reshape to [Batch, Channels, Signal_Length]
        self.Y = torch.from_numpy(self.Y).long().unsqueeze(1)  # Shape [Batch, 1]

        print(f"Loaded PTB 2 channels dataset with {len(self.X)} samples")

        assert (
            self.X.shape[1] == 2
        ), f"Expected 2 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

        if self.X.shape[2] < 250:
            self.X = F.pad(self.X, (0, 250 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 250]
        #     self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*250]
        elif self.X.shape[2] == 250:
            pass
        # elif self.X.shape[2] == 250:
        #     self.X = self.X.flatten(start_dim=1)  # Shape [Batch, Channels*250]
        else:
            raise ValueError(
                f"Expected signal length of 250, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # ds = torch.cat((self.X[index], self.Y[index]), dim=-1)  # Shape [Batch, Channels*250+1]
        return self.X[index], self.Y[index]


class PTB3ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root

        self.X = np.load(os.path.join(root, f"{split}.npy"))
        # Subsample time series
        indices = range(len(self.X))
        sub_indices = np.random.choice(indices, size=int(len(self.X) * 0.3), replace=False)
        self.X = self.X[sub_indices]
        self.Y = np.load(os.path.join(root, f"{split}_label.npy"))
        self.Y = self.Y[sub_indices]
        self.X = torch.from_numpy(self.X).float()
        self.X = self.X.reshape(self.X.shape[0], 3, -1)  # Reshape to [Batch, Channels, Signal_Length]
        self.Y = torch.from_numpy(self.Y).long().unsqueeze(1)  # Shape [Batch, 1]

        print(f"Loaded PTB 3 channels dataset with {len(self.X)} samples")
        assert (
            self.X.shape[1] == 3
        ), f"Expected 3 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

        if self.X.shape[2] < 166:
            self.X = F.pad(self.X, (0, 166 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 166]
        #     self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*166]
        elif self.X.shape[2] == 166:
            pass
        # elif self.X.shape[2] == 166:
        #     self.X = self.X.flatten(start_dim=1)  # Shape [Batch, Channels*166]
        else:
            raise ValueError(
                f"Expected signal length of 166, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # ds = torch.cat((self.X[index], self.Y[index]), dim=-1)  # Shape [Batch, Channels*166+1]
        return self.X[index], self.Y[index]


class PTB4ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root

        self.X = np.load(os.path.join(root, f"{split}.npy"))
        # Subsample time series
        indices = range(len(self.X))
        sub_indices = np.random.choice(indices, size=int(len(self.X) * 0.3), replace=False)
        self.X = self.X[sub_indices]
        self.Y = np.load(os.path.join(root, f"{split}_label.npy"))
        self.Y = self.Y[sub_indices]
        self.X = torch.from_numpy(self.X).float()
        self.X = self.X.reshape(self.X.shape[0], 4, -1)  # Reshape to [Batch, Channels, Signal_Length]
        self.Y = torch.from_numpy(self.Y).long().unsqueeze(1)  # Shape [Batch, 1]

        print(f"Loaded PTB 4 channels dataset with {len(self.X)} samples")
        assert (
            self.X.shape[1] == 4
        ), f"Expected 4 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

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


class PTB5ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root

        self.X = np.load(os.path.join(root, f"{split}.npy"))
        # Subsample time series
        indices = range(len(self.X))
        sub_indices = np.random.choice(indices, size=int(len(self.X) * 0.3), replace=False)
        self.X = self.X[sub_indices]
        self.Y = np.load(os.path.join(root, f"{split}_label.npy"))
        self.Y = self.Y[sub_indices]
        self.X = torch.from_numpy(self.X).float()
        self.X = self.X.reshape(self.X.shape[0], 5, -1)  # Reshape to [Batch, Channels, Signal_Length]
        self.Y = torch.from_numpy(self.Y).long().unsqueeze(1)  # Shape [Batch, 1]

        print(f"Loaded PTB 5 channels dataset with {len(self.X)} samples")
        assert (
            self.X.shape[1] == 5
        ), f"Expected 5 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

        if self.X.shape[2] < 100:
            self.X = F.pad(self.X, (0, 100 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 100]
        elif self.X.shape[2] == 100:
            pass
        else:
            raise ValueError(
                f"Expected signal length of 100, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class TUAB2ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.pkl"))

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for TUAB 2 channels...")
        for f in tqdm(self.files):
            with open(os.path.join(self.root, f), "rb") as rb:
                sample = pickle.load(rb)
                all_x.append(torch.from_numpy(sample["X"]).float())
                all_y.append(sample["y"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long).unsqueeze(1)  # Shape [Batch, 1]

        assert (
            self.X.shape[1] == 2
        ), f"Expected 2 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

        if self.X.shape[2] < 250:
            self.X = F.pad(self.X, (0, 250 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 250]
        #     self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*250]
        elif self.X.shape[2] == 250:
            pass
        # elif self.X.shape[2] == 250:
        #     self.X = self.X.flatten(start_dim=1)  # Shape [Batch, Channels*250]
        else:
            raise ValueError(
                f"Expected signal length of 250, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # print(f"Index: {index}, X shape: {self.X[index].shape}, Y shape: {self.Y[index].shape}")
        # ds = torch.cat((self.X[index], self.Y[index]), dim=-1)  # Shape [Batch, Channels, 250+1]
        return self.X[index], self.Y[index]


class TUAB3ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.pkl"))

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for TUAB 3 channels...")
        for f in tqdm(self.files):
            with open(os.path.join(self.root, f), "rb") as rb:
                sample = pickle.load(rb)
                all_x.append(torch.from_numpy(sample["X"]).float())
                all_y.append(sample["y"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long).unsqueeze(1)  # Shape [Batch, 1]

        assert (
            self.X.shape[1] == 3
        ), f"Expected 3 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

        if self.X.shape[2] < 166:
            self.X = F.pad(self.X, (0, 166 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 166]
        #     self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*166]
        # elif self.X.shape[2] == 166:
        #     self.X = self.X.flatten(start_dim=1)  # Shape [Batch, Channels*166]
        elif self.X.shape[2] == 166:
            pass
        else:
            raise ValueError(
                f"Expected signal length of 166, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # print(f"Index: {index}, X shape: {self.X[index].shape}, Y shape: {self.Y[index].shape}")
        return self.X[index], self.Y[index]


class TUAB4ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.pkl"))

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for TUAB 4 channels...")
        for f in tqdm(self.files):
            with open(os.path.join(self.root, f), "rb") as rb:
                sample = pickle.load(rb)
                all_x.append(torch.from_numpy(sample["X"]).float())
                all_y.append(sample["y"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long).unsqueeze(1)  # Shape [Batch, 1]

        assert (
            self.X.shape[1] == 4
        ), f"Expected 4 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

        if self.X.shape[2] < 125:
            self.X = F.pad(self.X, (0, 125 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 125]
        #     self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*125]
        # elif self.X.shape[2] == 125:
        #     self.X = self.X.flatten(start_dim=1)  # Shape [Batch, Channels*125]
        elif self.X.shape[2] == 125:
            pass
        else:
            raise ValueError(
                f"Expected signal length of 125, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # print(f"Index: {index}, X shape: {self.X[index].shape}, Y shape: {self.Y[index].shape}")
        return self.X[index], self.Y[index]


class TUAB5ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.pkl"))

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for TUAB 5 channels...")
        for f in tqdm(self.files):
            with open(os.path.join(self.root, f), "rb") as rb:
                sample = pickle.load(rb)
                all_x.append(torch.from_numpy(sample["X"]).float())
                all_y.append(sample["y"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long).unsqueeze(1)  # Shape [Batch, 1]

        assert (
            self.X.shape[1] == 5
        ), f"Expected 5 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

        if self.X.shape[2] < 100:
            self.X = F.pad(self.X, (0, 100 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 100]
        #     self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*100]
        # elif self.X.shape[2] == 100:
        #     self.X = self.X.flatten(start_dim=1)  # Shape [Batch, Channels*100]
        elif self.X.shape[2] == 100:
            pass
        else:
            raise ValueError(
                f"Expected signal length of 100, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # print(f"Index: {index}, X shape: {self.X[index].shape}, Y shape: {self.Y[index].shape}")
        return self.X[index], self.Y[index]


class TUEV2ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.pkl"))

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for TUEV 2 channels...")
        for f in tqdm(self.files):
            with open(os.path.join(self.root, f), "rb") as rb:
                sample = pickle.load(rb)
                all_x.append(torch.from_numpy(sample["signal"]).float())
                all_y.append(sample["label"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long) - 1  # Convert labels from 1-6 to 0-5
        self.Y = self.Y.unsqueeze(1)  # Shape [Batch, 1]

        assert (
            self.X.shape[1] == 2
        ), f"Expected 2 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

        if self.X.shape[2] < 250:
            self.X = F.pad(self.X, (0, 250 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 250]
        #     self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*250]
        elif self.X.shape[2] == 250:
            pass
        # elif self.X.shape[2] == 250:
        #     self.X = self.X.flatten(start_dim=1)  # Shape [Batch, Channels*250]
        else:
            raise ValueError(
                f"Expected signal length of 250, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # print(f"Index: {index}, X shape: {self.X[index].shape}, Y shape: {self.Y[index].shape}")
        return self.X[index], self.Y[index]


class TUEV3ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.pkl"))

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for TUEV 3 channels...")
        for f in tqdm(self.files):
            with open(os.path.join(self.root, f), "rb") as rb:
                sample = pickle.load(rb)
                all_x.append(torch.from_numpy(sample["signal"]).float())
                all_y.append(sample["label"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long) - 1  # Convert labels from 1-6 to 0-5
        self.Y = self.Y.unsqueeze(1)  # Shape [Batch, 1]

        assert (
            self.X.shape[1] == 3
        ), f"Expected 3 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

        if self.X.shape[2] < 166:
            self.X = F.pad(self.X, (0, 166 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 166]
        #     self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*166]
        elif self.X.shape[2] == 166:
            pass
        # elif self.X.shape[2] == 166:
        #     self.X = self.X.flatten(start_dim=1)  # Shape [Batch, Channels*166]
        else:
            raise ValueError(
                f"Expected signal length of 166, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # print(f"Index: {index}, X shape: {self.X[index].shape}, Y shape: {self.Y[index].shape}")
        return self.X[index], self.Y[index]


class TUEV4ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.pkl"))

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for TUEV 4 channels...")
        for f in tqdm(self.files):
            with open(os.path.join(self.root, f), "rb") as rb:
                sample = pickle.load(rb)
                all_x.append(torch.from_numpy(sample["signal"]).float())
                all_y.append(sample["label"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long) - 1  # Convert labels from 1-6 to 0-5
        self.Y = self.Y.unsqueeze(1)  # Shape [Batch, 1]

        assert (
            self.X.shape[1] == 4
        ), f"Expected 4 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

        if self.X.shape[2] < 125:
            self.X = F.pad(self.X, (0, 125 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 125]
        #     self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*125]
        elif self.X.shape[2] == 125:
            pass
        # elif self.X.shape[2] == 125:
        #     self.X = self.X.flatten(start_dim=1)  # Shape [Batch, Channels*125]
        else:
            raise ValueError(
                f"Expected signal length of 125, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # print(f"Index: {index}, X shape: {self.X[index].shape}, Y shape: {self.Y[index].shape}")
        return self.X[index], self.Y[index]


class TUEV5ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.pkl"))

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for TUEV 5 channels...")
        for f in tqdm(self.files):
            with open(os.path.join(self.root, f), "rb") as rb:
                sample = pickle.load(rb)
                all_x.append(torch.from_numpy(sample["signal"]).float())
                all_y.append(sample["label"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long) - 1  # Convert labels from 1-6 to 0-5
        self.Y = self.Y.unsqueeze(1)  # Shape [Batch, 1]

        assert (
            self.X.shape[1] == 5
        ), f"Expected 5 channels, but got {self.X.shape[1]}. Please check the data preprocessing."

        if self.X.shape[2] < 100:
            self.X = F.pad(self.X, (0, 100 - self.X.shape[2]), "constant", 0)  # New shape [Batch, Channels, 100]
        #     self.X = self.X.flatten(start_dim=1)  # New shape [Batch, Channels*100]
        elif self.X.shape[2] == 100:
            pass
        # elif self.X.shape[2] == 100:
        #     self.X = self.X.flatten(start_dim=1)  # Shape [Batch, Channels*100]
        else:
            raise ValueError(
                f"Expected signal length of 100, but got {self.X.shape[2]}. Please check the data preprocessing."
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # print(f"Index: {index}, X shape: {self.X[index].shape}, Y shape: {self.Y[index].shape}")
        return self.X[index], self.Y[index]


class HIRID2ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.npy"))
        self.label_directory = os.path.dirname(self.root)
        self.labels = pd.read_csv(os.path.join(self.label_directory, "labels.csv"), index_col="patientid")

        label_map = {"alive": 0, "dead": 1}
        self.labels["discharge_status"] = self.labels["discharge_status"].map(label_map)

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for HIRID 2 channels...")
        for f in tqdm(self.files):
            sample = np.load(os.path.join(self.root, f))
            all_x.append(torch.from_numpy(sample).float().transpose(-1, -2))  # Transpose to [Channels, Time]
            patient_id = os.path.basename(f).replace(".npy", "")
            all_y.append(self.labels.loc[int(patient_id), "discharge_status"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long).unsqueeze(1)  # Shape [Batch, 1]

        print(f"X shape: {self.X.shape}, Y shape: {self.Y.shape}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class HIRID3ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.npy"))
        self.label_directory = os.path.dirname(self.root)
        self.labels = pd.read_csv(os.path.join(self.label_directory, "labels.csv"), index_col="patientid")

        label_map = {"alive": 0, "dead": 1}
        self.labels["discharge_status"] = self.labels["discharge_status"].map(label_map)

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for HIRID 3 channels...")
        for f in tqdm(self.files):
            sample = np.load(os.path.join(self.root, f))
            all_x.append(torch.from_numpy(sample).float().transpose(-1, -2))  # Transpose to [Channels, Time]
            patient_id = os.path.basename(f).replace(".npy", "")
            all_y.append(self.labels.loc[int(patient_id), "discharge_status"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long).unsqueeze(1)  # Shape [Batch, 1]

        print(f"X shape: {self.X.shape}, Y shape: {self.Y.shape}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class HIRID4ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.npy"))
        self.label_directory = os.path.dirname(self.root)
        self.labels = pd.read_csv(os.path.join(self.label_directory, "labels.csv"), index_col="patientid")

        label_map = {"alive": 0, "dead": 1}
        self.labels["discharge_status"] = self.labels["discharge_status"].map(label_map)

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for HIRID 4 channels...")
        for f in tqdm(self.files):
            sample = np.load(os.path.join(self.root, f))
            all_x.append(torch.from_numpy(sample).float().transpose(-1, -2))  # Transpose to [Channels, Time]
            patient_id = os.path.basename(f).replace(".npy", "")
            all_y.append(self.labels.loc[int(patient_id), "discharge_status"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long).unsqueeze(1)  # Shape [Batch, 1]

        print(f"X shape: {self.X.shape}, Y shape: {self.Y.shape}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class HIRID5ChannelDataset(Dataset):
    def __init__(self, root, split):
        self.root = root
        self.files = glob(os.path.join(root, f"{split}/*.npy"))
        self.label_directory = os.path.dirname(self.root)
        self.labels = pd.read_csv(os.path.join(self.label_directory, "labels.csv"), index_col="patientid")

        label_map = {"alive": 0, "dead": 1}
        self.labels["discharge_status"] = self.labels["discharge_status"].map(label_map)

        all_x = []
        all_y = []

        print(f"Loading {len(self.files)} samples into RAM for HIRID 5 channels...")
        for f in tqdm(self.files):
            sample = np.load(os.path.join(self.root, f))
            all_x.append(torch.from_numpy(sample).float().transpose(-1, -2))  # Transpose to [Channels, Time]
            patient_id = os.path.basename(f).replace(".npy", "")
            all_y.append(self.labels.loc[int(patient_id), "discharge_status"])

        self.X = torch.stack(all_x)
        self.Y = torch.tensor(all_y, dtype=torch.long).unsqueeze(1)  # Shape [Batch, 1]

        print(f"X shape: {self.X.shape}, Y shape: {self.Y.shape}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class TSPFNMetaDataset(Dataset):
    def __init__(self, datasets: Dict, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        self.chunks = []

        for dataset in datasets.values():
            n = len(dataset.X)
            if n < chunk_size:
                # Optionnel : On peut ignorer ou padder les datasets trop petits
                raise ValueError(
                    f"Dataset of size {n} is smaller than chunk size {chunk_size}. Please check the datasets or adjust the chunk size."
                )

            for i in range(0, n - chunk_size + 1, chunk_size):
                # self.chunks.append((X[i : i + chunk_size], y[i : i + chunk_size]))
                self.chunks.append((dataset[i : i + chunk_size]))

            # (Overlapping last chunk)
            if n % chunk_size != 0:
                # self.chunks.append((X[-chunk_size:], y[-chunk_size:]))
                self.chunks.append((dataset[-chunk_size:]))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        # On retourne le bloc de 10k (X et y)
        # Supposons que y est la dernière colonne
        x, y = self.chunks[idx]
        return x, y


class TSPFNValidationDataset(Dataset):
    def __init__(self, train_datasets_list: Dict, val_datasets_list: Dict, chunk_size=10000):
        # self.n_support = n_support
        # self.n_query = n_query
        self.pairs = []

        self.n_support = int(chunk_size * 0.8)  # 80% for support
        self.n_query = chunk_size - self.n_support  # 20% for query

        for (train_dataset), (val_dataset) in zip(train_datasets_list.values(), val_datasets_list.values()):
            # 1. On découpe le Val en chunks de 2000 (les Query)
            n_v = len(val_dataset.X)
            # On utilise le sliding window pour ne rien perdre du Val
            indices = range(0, n_v - self.n_query + 1, self.n_query)

            for i in indices:
                query_chunk, label_chunk = val_dataset[i : i + self.n_query]
                # On stocke le chunk de val ET une référence au train complet associé
                self.pairs.append(
                    {"full_train": (train_dataset.X, train_dataset.Y), "query_chunk": (query_chunk, label_chunk)}
                )

            # (Overlapping last chunk for Val)
            if n_v % self.n_query != 0:
                self.pairs.append(
                    {
                        "full_train": (train_dataset.X, train_dataset.Y),
                        "query_chunk": (val_dataset.X[-self.n_query :], val_dataset.Y[-self.n_query :]),
                    }
                )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        train_data, train_labels = item["full_train"]
        query_chunk, query_labels = item["query_chunk"]

        # Random selection of support set from the full train data
        indices_sup = torch.randperm(len(train_data))[: self.n_support]
        support_chunk = train_data[indices_sup]
        support_labels = train_labels[indices_sup]

        output = {"support": (support_chunk, support_labels), "query": (query_chunk, query_labels)}

        return output


class TSPFNTestDataset(Dataset):
    def __init__(self, train_datasets_list: Dict, test_datasets_list: Dict, chunk_size=10000):
        self.n_support = int(chunk_size * 0.8)  # 80% for support
        self.n_query = chunk_size - self.n_support  # 20% for query
        self.pairs = []

        for (train_dataset), (test_dataset) in zip(train_datasets_list.values(), test_datasets_list.values()):
            # 1. On découpe le Val en chunks de 2000 (les Query)
            n_v = len(test_dataset.X)
            # On utilise le sliding window pour ne rien perdre du Val
            indices = range(0, n_v - self.n_query + 1, self.n_query)
            for i in indices:
                query_chunk, label_chunk = test_dataset[i : i + self.n_query]
                # On stocke le chunk de val ET une référence au train complet associé
                self.pairs.append(
                    {"full_train": (train_dataset.X, train_dataset.Y), "query_chunk": (query_chunk, label_chunk)}
                )

            # (Overlapping last chunk for Val)
            if n_v % self.n_query != 0:
                self.pairs.append(
                    {
                        "full_train": (train_dataset.X, train_dataset.Y),
                        "query_chunk": (test_dataset.X[-self.n_query :], test_dataset.Y[-self.n_query :]),
                    }
                )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        train_data, train_labels = item["full_train"]
        query_chunk, query_labels = item["query_chunk"]

        # Random selection of support set from the full train data
        indices_sup = torch.randperm(len(train_data))[: self.n_support]
        support_chunk = train_data[indices_sup]
        support_labels = train_labels[indices_sup]

        output = {"support": (support_chunk, support_labels), "query": (query_chunk, query_labels)}

        return output

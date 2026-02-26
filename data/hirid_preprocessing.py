import pandas as pd
import csv
import gzip
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

ORIGIN_DIRECTORY = "/path/to/folder/HIRID/data"
OUTPUT_DIRECTORY = ORIGIN_DIRECTORY
PATH_TO_LABELS = "/path/to/folder/HIRID/labels.csv"
KEEP_5CHANNELS = ["patientid", "reldatetime", "vm1", "pm41", "vm5", "vm20", "vm62"]
KEEP_4CHANNELS = ["patientid", "reldatetime", "vm1", "pm41", "vm5", "vm20", "vm62"]
KEEP_3CHANNELS = ["patientid", "reldatetime", "vm1", "pm41", "vm5"]
KEEP_2CHANNELS = ["patientid", "reldatetime", "vm1", "pm41"]
WINDOW_SIZE = 100


def preprocess_hirid_data(
    origin_dir,
    output_dir,
    keep_channels,
    window_size,
    path_to_labels = PATH_TO_LABELS,
):
    labels_df = pd.read_csv(path_to_labels, index_col="patientid")
    for file in tqdm(
        glob(os.path.join(origin_dir, "*.csv")),
        total=len(glob(os.path.join(origin_dir, "*.csv"))),
        desc="Preprocessing HIRID CSV files",
    ):
        with open(file, "r") as f:
            df = pd.read_csv(f)
            # Keep only pids in label df
            df = df[df["patientid"].isin(labels_df.index)]
        df_group = df.groupby(["patientid", "reldatetime"])[keep_channels[2:]].mean()
        patient_matrices = []
        for patient_id, group in df_group.groupby(level=0):
            vals = group.values
            num_steps = vals.shape[0]

            if num_steps < window_size:
                # Identify the last recorded values for this specific patient
                last_val = vals[-1, :]

                # Create a padding matrix where every row is that last recorded value
                padding_rows = window_size - num_steps
                padding = np.tile(last_val, (padding_rows, 1))

                # Stack the real data on top of the 'edge' padding
                padded_group = np.vstack([vals, padding])
                patient_matrices.append(padded_group)
            else:
                # If the stay is longer than window_size, we take the first window_size steps
                # This aligns with SOTA 'First 24h' windowing logic
                padded_group = vals[:window_size, :]
                patient_matrices.append(padded_group)

            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{patient_id}.npy")
            np.save(output_file, np.array(padded_group))


def split_hirid_data(output_dir, train_ratio=0.8, seed=42, labels_path=PATH_TO_LABELS):
    np.random.seed(seed)
    labels = pd.read_csv(labels_path, index_col="patientid")
    indices = list(labels.index)
    train_indices, test_indices = train_test_split(
        indices, test_size=1 - train_ratio, random_state=seed, stratify=labels["discharge_status"]
    )
    print(f"Train indices: {train_indices[:10]}...")  # Print first 10 indices for verification
    print(f"Val indices: {test_indices[:10]}...")  # Print first 10 indices for verification

    for split, split_indices in zip(["train", "val"], [train_indices, test_indices]):
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for idx in tqdm(split_indices, total=len(split_indices), desc=f"Saving {split} data"):
            patient_id = idx  # Assuming patient_id is the same as the index in labels.csv
            input_file = os.path.join(output_dir, f"{patient_id}.npy")
            output_file = os.path.join(split_dir, f"{patient_id}.npy")
            if os.path.exists(input_file):
                os.rename(input_file, output_file)
            else:
                print(f"Warning: File {input_file} does not exist and will be skipped.")


if __name__ == "__main__":
    preprocess_hirid_data(
        origin_dir=ORIGIN_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY + "/fivechannels",
        keep_channels=KEEP_5CHANNELS,
        window_size=100,
        path_to_labels=PATH_TO_LABELS,
    )
    split_hirid_data(
        output_dir=OUTPUT_DIRECTORY + "/fivechannels", train_ratio=0.8, seed=42, labels_path=PATH_TO_LABELS
    )
    preprocess_hirid_data(
        origin_dir=ORIGIN_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY + "/fourchannels",
        keep_channels=KEEP_4CHANNELS,
        window_size=125,
        path_to_labels=PATH_TO_LABELS,
    )
    split_hirid_data(
        output_dir=OUTPUT_DIRECTORY + "/fourchannels", train_ratio=0.8, seed=42, labels_path=PATH_TO_LABELS
    )
    preprocess_hirid_data(
        origin_dir=ORIGIN_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY + "/threechannels",
        keep_channels=KEEP_3CHANNELS,
        window_size=166,
        path_to_labels=PATH_TO_LABELS,
    )
    split_hirid_data(
        output_dir=OUTPUT_DIRECTORY + "/threechannels", train_ratio=0.8, seed=42, labels_path=PATH_TO_LABELS
    )
    preprocess_hirid_data(
        origin_dir=ORIGIN_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY + "/twochannels",
        keep_channels=KEEP_2CHANNELS,
        window_size=250,
        path_to_labels=PATH_TO_LABELS,
    )
    split_hirid_data(output_dir=OUTPUT_DIRECTORY + "/twochannels", train_ratio=0.8, seed=42, labels_path=PATH_TO_LABELS)

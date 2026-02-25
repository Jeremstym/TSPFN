import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from typing import List, Union
from tqdm import tqdm


def load_csv(subset_path: Path, split_ratio: float) -> None:
    label_encoder = LabelEncoder()
    path = os.path.join(subset_path)
    name_csv = os.path.basename(path)
    assert os.path.isfile(path), f"Dataset file not found: {path}"
    # Get number of lines in the file
    with open(path, "r") as f:
        total_lines = sum(1 for _ in f)

    list_df = []
    with tqdm(total=total_lines, desc=f"Loading {name_csv}") as pbar:
        for chunk in pd.read_csv(path, chunksize=1000, index_col=0):
            list_df.append(chunk)
            pbar.update(chunk.shape[0])

    df = pd.concat(list_df, ignore_index=False)
    # Encode labels to integers
    df_label = label_encoder.fit_transform(df.iloc[:, -1])
    df_features = df.iloc[:, :-1]

    df_values = df_features.values
    if df_values.shape[1] < 500:
        # Pad with zeros to have consistent feature size
        padding = np.zeros((df_values.shape[0], 500 - df_values.shape[1]))
        df_values = np.hstack((df_values, padding))
    elif df_values.shape[1] > 500:
        # Truncate to 500 features
        df_values = df_values[:, :500]
    df_values = np.hstack((df_values, df_label.reshape(-1, 1)))

    # Split dataset
    indices = np.arange(len(df_values))
    labels = df_values[:, -1]
    try:
        train_indices, val_indices = train_test_split(
            indices, train_size=split_ratio, random_state=42, shuffle=True, stratify=labels
        )
    except ValueError:
        print(f"Warning: Stratified split failed for {name_csv}, using non-stratified split instead.")
        train_indices, val_indices = train_test_split(
            indices, train_size=split_ratio, random_state=42, shuffle=True, stratify=None
        )
    df_train = df_values[train_indices]

    data_train_ts = []
    data_val_ts = []
    if df_train.shape[0] // 1024 > 1:
        # Split into chunks of 1024 samples
        chunk_size = 1024
        chunk_val_size = 64
        usable_size = (df_train.shape[0] // chunk_size) * chunk_size
        data_chunked = df_train[:usable_size]
        data_chunked = data_chunked.reshape(-1, chunk_size, df_train.shape[1])
        for chunk in data_chunked:
            data_support, data_query = train_test_split(
                chunk, test_size=0.5, random_state=42, shuffle=True, stratify=chunk[:, -1]
            )
            data_chunk = np.concatenate([data_support, data_query], axis=0)
            data_train_ts.append(torch.tensor(data_chunk, dtype=torch.float32))
            data_val_chunk_indices = np.random.choice(val_indices, size=chunk_val_size, replace=False)
            data_val_chunk = df_values[data_val_chunk_indices]
            data_val_ts.append(torch.tensor(data_val_chunk, dtype=torch.float32))
    else:
        data_train_ts = []
        data_val_ts = []

    return data_train_ts, data_val_ts
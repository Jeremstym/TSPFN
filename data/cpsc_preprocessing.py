import pickle
from multiprocessing import Pool
import scipy.signal as sgn
from scipy.signal import resample
import torch.nn.functional as F
import pandas as pd
import numpy as np
import wfdb
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import os
import sys

from data.utils_ptb.segment_utils import find_rpeaks_clean_ecgs_in_dataframe, segment_ecg_in_clean_dataframe
from data.utils_ptb.segment_utils import values_from_dataframe_ny_list


sys.path.insert(0, os.path.abspath(".."))
rng = np.random.default_rng(seed=42)

# CHOSEN_CHANNELS = [0, 1, 2, 3, 4]  # Fix channel
# CHOSEN_CHANNELS = [0, 1, 2, 3]  # Fix channel
CHOSEN_CHANNELS = [0, 1, 2]  # Fix channel


def group_cpsc_labels(labels, mode="5_cat"):
    """
    Original Mapping (assuming standard order):
    0: 164884008 (STD)
    1: 164889003 (AFib)
    2: 164909002 (LBBB)
    3: 164931005 (STE)
    4: 270492004 (IAVB)
    5: 284470004 (PAC)
    6: 426783006 (Normal)
    7: 429622005 (PVC)
    8: 59118001  (RBBB)
    """

    # 0: Normal, 1: AFib, 2: Other Arrhythmia, 3: Conduction Block, 4: ST-Change
    map_5 = {
        6: 0,  # Normal
        1: 1,  # AFib
        5: 2,
        7: 2,  # PAC, PVC -> Other Arrhythmia
        2: 3,
        4: 3,
        8: 3,  # LBBB, IAVB, RBBB -> Blocks
        0: 4,
        3: 4,  # STD, STE -> ST-Changes
    }

    # 0: Normal, 1: Arrhythmias (inc. AFib), 2: Blocks, 3: ST-Change
    map_4 = {
        6: 0,  # Normal
        1: 1,
        5: 1,
        7: 1,  # AFib, PAC, PVC -> Arrhythmias
        2: 2,
        4: 2,
        8: 2,  # LBBB, IAVB, RBBB -> Blocks
        0: 3,
        3: 3,  # STD, STE -> ST-Changes
    }

    mapping = map_5 if mode == "5_cat" else map_4

    # Vectorized mapping
    new_labels = np.array([mapping[label] for label in labels])
    return new_labels


def resample_hb_batch(data, fs_in, fs_out):
    """
    data shape: (N, 400, 2) -> (N, Time, Channels)
    """
    L_original = data.shape[1]
    L_out = int(np.round(L_original * fs_out / fs_in))

    # Resample along axis 1 (the 400-sample time dimension)
    # This preserves axis 0 (Batch) and axis 2 (Channels)
    resampled_data = resample(data, L_out, axis=1)

    return resampled_data


def load_raw_data(df, path):

    # data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = []
    file_name_kept = []
    for f in tqdm(df.file_name, total=len(df)):
        record = wfdb.rdsamp(path + f)
        if record[0].shape[0] <= 5000:
            # print(f"Record {f} has less than 5000 samples, skipping.")
            continue
        elif record[0].shape[0] > 5000:
            new_signal = record[0][:5000, :]
            new_record = (new_signal, record[1])
            data.append(new_record)
            file_name_kept.append(f)
        else:
            data.append(record)
            file_name_kept.append(f)
    data = np.array([signal for signal, meta in data])
    df = df[df.file_name.isin(file_name_kept)].reset_index(drop=True)
    return data, df


def standardize(row):
    signal = row.ecg_signal_raw
    mean = row.mean_ecg
    std = row.std_ecg
    return (signal - mean) / std


def standardize_hb(row):
    mean = row.mean_ecg
    std = row.std_ecg
    hb = row.ecg_signal_heartbeat
    if isinstance(hb, list):
        hb = np.array(hb)
    # print(hb, mean)
    return (hb - mean) / std


def get_values(signals):
    signals_arr = np.zeros([len(signals), signals[0].shape[0]])
    for i in range(len(signals)):
        signals_arr[i, :] = signals[i]
    return signals_arr


def extract_hb_from_dataframe(df):

    heart_beats, len_heart_beats = values_from_dataframe_ny_list(df, "ecg_signal_heartbeat", as_list=True)
    heart_beats_indexes, _ = values_from_dataframe_ny_list(df, "heartbeat_indexes", as_list=True)
    true_labels_ = df.true_label.values[:]
    true_labels = np.array([item for item, count in zip(true_labels_, len_heart_beats) for _ in range(count)])
    heart_beats = np.vstack(heart_beats)
    heart_beats_indexes = np.vstack(heart_beats_indexes)
    return heart_beats, true_labels


def count_occurrences(a):
    unique, counts = np.unique(a, return_counts=True)
    return dict(zip(unique, counts))


if __name__ == "__main__":

    path = "/data/stympopper/CARDIOLOGY/cpsc_2018/"
    path_data = "/data/stympopper/CARDIOLOGY/cpsc_2018/data/"
    # sampling_rate = 500
    resampled_rate = 200

    # load and convert annotation data
    Y = pd.read_csv(path + "cpsc_2018_labels.csv")
    Y["label"] = group_cpsc_labels(Y["label"], mode="4_cat")  # Map to 4 categories (Normal, Arrhythmia, Block, ST-Change)
    # Load raw signal data
    X, Y = load_raw_data(Y, path_data)
    X = resample_hb_batch(X, fs_in=500, fs_out=resampled_rate)

    # Split data into train and test
    indices = np.arange(len(Y))
    _, subsample_indices = train_test_split(indices, test_size=1000, random_state=42, stratify=Y["label"])
    train_indices, test_indices = train_test_split(
        subsample_indices, test_size=0.2, random_state=42, stratify=Y.iloc[subsample_indices]["label"]
    )
    # Train
    X_train = X[train_indices]
    y_train = Y.iloc[train_indices]
    # Test
    X_test = X[test_indices]
    y_test = Y.iloc[test_indices]

    y_train = pd.Series(y_train["label"].values, name="true_label")
    y_test = pd.Series(y_test["label"].values, name="true_label")

    X_test = X_test[:, :, CHOSEN_CHANNELS]
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

    X_train = X_train[:, :, CHOSEN_CHANNELS]
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")

    print(f"y_test.value_counts(sort=False): {y_test.value_counts(sort=False)}")

    print(f"y_train.value_counts(sort=False): {y_train.value_counts(sort=False)}")

    # Window data by heart beats and save
    list_of_arrays_train = [row for row in X_train]
    list_of_arrays_test = [row for row in X_test]

    df_train = pd.DataFrame(
        {"ecg_signal_raw": list_of_arrays_train, "true_label": y_train, "partition": ["train"] * len(y_train)}
    )
    df_test = pd.DataFrame(
        {"ecg_signal_raw": list_of_arrays_test, "true_label": y_test, "partition": ["test"] * len(y_test)}
    )

    df = pd.concat([df_train, df_test])
    print(f"dataset shape before finding R-peaks: {df.shape}")
    df.to_pickle(path + "cpsc_dataframe.pkl")

    print("Finding R-peaks in the ECG signals...")
    df_rp = find_rpeaks_clean_ecgs_in_dataframe(data=df, rate=resampled_rate)
    print("R-peaks found and added to the dataframe.")
    df_rp.to_pickle(path + "cpsc_dataframe_rp.pkl")
    df_rp = pd.read_pickle(path + "cpsc_dataframe_rp.pkl")
    print(f"dataset shape after finding R-peaks: {df_rp.shape}")

    print("Segmenting ECG signals into heartbeats...")
    df_final = segment_ecg_in_clean_dataframe(
        ROOT=path, data=df_rp, size_before_index=130, size_after_index=270, signal_length=2000
    )
    print("ECG signals segmented into heartbeats and added to the dataframe.")

    df_final.to_pickle(path + "cpsc_dataframe_final.pkl")

    df_final = pd.read_pickle(path + "cpsc_dataframe_final.pkl")
    print(f"dataset shape after segmenting into heartbeats: {df_final.shape}")

    train_data = df_final[df_final["partition"] == "train"]
    test_data = df_final[df_final["partition"] == "test"]
    # Filter out rows where no heartbeats were found
    train = train_data[train_data["ecg_signal_heartbeat"].map(len) > 0].copy()
    test = test_data[test_data["ecg_signal_heartbeat"].map(len) > 0].copy()

    print("Extracting heartbeats and labels for train and test sets...")
    X_train, y_train = extract_hb_from_dataframe(train)
    X_test, y_test = extract_hb_from_dataframe(test)
    print("Heartbeats and labels extracted for train and test sets.")

    print(f"Downsampled heartbeats shape - Train: {X_train.shape}, Test: {X_test.shape}")
    X_train = resample_hb_batch(X_train, fs_in=400, fs_out=166)
    X_test = resample_hb_batch(X_test, fs_in=400, fs_out=166)
    print(f"Downsampled heartbeats shape - Train: {X_train.shape}, Test: {X_test.shape}")

    print(f"Flatten on channel dimension - Train: {X_train.shape}, Test: {X_test.shape}")
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print(f"Flattened heartbeats shape - Train: {X_train.shape}, Test: {X_test.shape}")

    print("TRAIN")
    print(X_train.shape, y_train.shape)
    print(count_occurrences(y_train))

    print("TEST")
    print(X_test.shape, y_test.shape)
    print(count_occurrences(y_test))

    target_path = path + "threechannels/"
    os.makedirs(target_path, exist_ok=True)
    np.save(target_path + "train.npy", X_train)
    np.save(target_path + "val.npy", X_test)
    np.save(target_path + "train_label.npy", y_train)
    np.save(target_path + "val_label.npy", y_test)

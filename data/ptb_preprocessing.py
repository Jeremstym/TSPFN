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

CHOSEN_CHANNELS = [0, 1, 2, 3, 4]  # Fix channel


def resample_hb_batch(data, fs_in, fs_out):
    """
    data shape: (N, 500, 2) -> (N, Time, Channels)
    """
    L_original = data.shape[1]
    L_out = int(np.round(L_original * fs_out / fs_in))
    
    # Resample along axis 1 (the 500-sample time dimension)
    # This preserves axis 0 (Batch) and axis 2 (Channels)
    resampled_data = resample(data, L_out, axis=1)
    
    return resampled_data


def convert_labels(label):
    labels = {"MI": 1, "NORM": 0, "HYP": 1, "CD": 1, "STTC": 1}
    return labels[label]


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        # data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        data = []
        for f in tqdm(df.filename_lr, total=len(df)):
            record = wfdb.rdsamp(path + f)
            data.append(record)
    else:
        # data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
        data = []
        for f in tqdm(df.filename_hr, total=len(df)):
            record = wfdb.rdsamp(path + f)
            data.append(record)
    data = np.array([signal for signal, meta in data])
    return data


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

    path = "/data/stympopper/PTB/data/"
    sampling_rate = 500

    # load and convert annotation data
    Y = pd.read_csv(path + "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path + "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in tqdm(y_dic.keys(), ncols=70):
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # Apply diagnostic superclass
    Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

    # Split data into train and test
    test_fold = 10
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    index_ok_test = y_test.str.len() == 1
    index_ok_train = y_train.str.len() == 1

    y_test = y_test[index_ok_test]
    X_test = X_test[index_ok_test]
    # # Subsample
    # indices = np.arange(len(y_test))
    # test_selected_indices = rng.choice(indices, size=500, replace=False)
    # y_test = y_test.iloc[test_selected_indices]
    # X_test = X_test[test_selected_indices]
    y_test = y_test.apply(lambda x: x[0])
    y_test = y_test.apply(lambda x: convert_labels(x))
    print(X_test.shape, y_test.shape)
    X_test = X_test[:, :, CHOSEN_CHANNELS]

    y_train = y_train[index_ok_train]
    X_train = X_train[index_ok_train]
    # # Subsample
    # indices = np.arange(len(y_train))
    # train_selected_indices = rng.choice(indices, size=5000, replace=False)
    # y_train = y_train.iloc[train_selected_indices]
    # X_train = X_train[train_selected_indices]
    y_train = y_train.apply(lambda x: x[0])
    y_train = y_train.apply(lambda x: convert_labels(x))
    print(X_train.shape, y_train.shape)
    X_train = X_train[:, :, CHOSEN_CHANNELS]

    print(X_test.shape, y_test.shape)
    print(y_test.value_counts(sort=False))

    print(X_train.shape, y_train.shape)
    print(y_train.value_counts(sort=False))

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
    df.to_pickle(path + "ptbxl_dataframe.pkl")

    # if os.path.exists(path+"ptbxl_dataframe_rp.pkl"):
    #     df_rp = pd.read_pickle(path+"ptbxl_dataframe_rp.pkl")
    # else:
    print("Finding R-peaks in the ECG signals...")
    df_rp = find_rpeaks_clean_ecgs_in_dataframe(data=df)
    print("R-peaks found and added to the dataframe.")
    df_rp.to_pickle(path + "ptbxl_dataframe_rp.pkl")

    print("Segmenting ECG signals into heartbeats...")
    df_final = segment_ecg_in_clean_dataframe(ROOT=path, data=df_rp)
    print("ECG signals segmented into heartbeats and added to the dataframe.")
    # print(df_final.columns)
    # df_final["heartbeat_indexes"]

    # # print(df_final)
    df_final.to_pickle(path + "ptbxl_dataframe_final.pkl")

    df_final = pd.read_pickle(path + "ptbxl_dataframe_final.pkl")
    # df_final.head()

    # print("Standardizing ECG signals...")
    # df_final['mean_ecg'] = df_final.ecg_signal_raw.apply(np.mean)
    # df_final['std_ecg'] = df_final.ecg_signal_raw.apply(np.std)
    # df_final['ecg_standardize_signal'] = df_final.apply(standardize, axis=1)
    # df_final['ecg_standardize_signal_heartbeats'] = df_final.apply(standardize_hb, axis=1)
    # print("ECG signals standardized and added to the dataframe.")

    # df_final.to_pickle(path + "ptbxl_dataframe_final.pkl")
    # df_final.head()

    # features = get_values(df_final.ecg_standardize_signal.values[:])
    # y = df_final.true_label.values[:]
    # print(f"features shape: {features.shape}, y shape: {y.shape}")

    # print("Extracting heartbeats and labels from the dataframe...")
    # heart_beats, len_heart_beats = values_from_dataframe_ny_list(df_final, 'ecg_signal_heartbeat', as_list=True)
    # heart_beats_indexes, _ = values_from_dataframe_ny_list(df_final, 'heartbeat_indexes', as_list=True)
    # true_labels_ = df_final.true_label.values[:]
    # true_labels = np.array([item for item, count in zip(true_labels_, len_heart_beats) for _ in range(count)])
    # heart_beats = np.vstack(heart_beats)
    # heart_beats_indexes = np.vstack(heart_beats_indexes)
    # print("Heartbeats and labels extracted.")

    # print(f"heart_beats shape: {heart_beats.shape}, true_labels shape: {true_labels.shape}, heart_beats_indexes shape: {heart_beats_indexes.shape}")

    train_data = df_final[df_final["partition"] == "train"]
    test_data = df_final[df_final["partition"] == "test"]
    # train_data, validation_data, _, _ = train_test_split(train_data, train_data, test_size=0.2, random_state=42)
    # validation_data["partition"] = "valid"
    # df_updated = pd.concat([train_data, validation_data, test_data])
    # df_updated[df_updated['partition']=='train'].head()

    train = train_data
    test = test_data
    # valid = df_updated[df_updated.partition == "valid"]

    print("Extracting heartbeats and labels for train and test sets...")
    X_train, y_train = extract_hb_from_dataframe(train)
    # X_val, y_val = extract_hb_from_dataframe(valid)
    X_test, y_test = extract_hb_from_dataframe(test)
    print("Heartbeats and labels extracted for train and test sets.")

    print(f"Downsampled heartbeats shape - Train: {X_train.shape}, Test: {X_test.shape}")
    X_train = resample_hb_batch(X_train, fs_in=500, fs_out=100)
    # X_val = resample_hb_batch(X_val, fs_in=500, fs_out=100)
    X_test = resample_hb_batch(X_test, fs_in=500, fs_out=100)
    print(f"Downsampled heartbeats shape - Train: {X_train.shape}, Test: {X_test.shape}")

    print(f"Flatten on channel dimension - Train: {X_train.shape}, Test: {X_test.shape}")
    X_train = X_train.reshape(X_train.shape[0], -1)
    # X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print(f"Flattened heartbeats shape - Train: {X_train.shape}, Test: {X_test.shape}")

    print("TRAIN")
    print(X_train.shape, y_train.shape)
    print(count_occurrences(y_train))

    # print("VALID")
    # print(X_val.shape, y_val.shape)
    # print(count_occurrences(y_val))

    print("TEST")
    print(X_test.shape, y_test.shape)
    print(count_occurrences(y_test))

    target_path = path + "fivechannels/"
    os.makedirs(target_path, exist_ok=True)
    np.save(target_path + "train.npy", X_train)
    # np.save(target_path + "heldout.npy", X_val)
    np.save(target_path + "val.npy", X_test)
    np.save(target_path + "train_label.npy", y_train)
    # np.save(target_path + "heldout_label.npy", y_val)
    np.save(target_path + "val_label.npy", y_test)

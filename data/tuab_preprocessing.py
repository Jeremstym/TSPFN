# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI (2024)
# By Wei-Bang Jiang (2024)
# Based on BIOT code base
# https://github.com/ycq091044/BIOT
# --------------------------------------------------------
import os
import pickle

from multiprocessing import Pool
import numpy as np
import scipy.signal as sgn
import torch.nn.functional as F
import mne


drop_channels = [
    "PHOTIC-REF",
    "IBI",
    "BURSTS",
    "SUPPR",
    "EEG ROC-REF",
    "EEG LOC-REF",
    "EEG EKG1-REF",
    "EMG-REF",
    "EEG C3P-REF",
    "EEG C4P-REF",
    "EEG SP1-REF",
    "EEG SP2-REF",
    "EEG LUC-REF",
    "EEG RLC-REF",
    "EEG RESP1-REF",
    "EEG RESP2-REF",
    "EEG EKG-REF",
    "RESP ABDOMEN-REF",
    "ECG EKG-REF",
    "PULSE RATE",
    "EEG PG2-REF",
    "EEG PG1-REF",
]
drop_channels.extend([f"EEG {i}-REF" for i in range(20, 129)])
chOrder_standard = [
    "EEG FP1-REF",
    "EEG FP2-REF",
    "EEG F3-REF",
    "EEG F4-REF",
    "EEG C3-REF",
    "EEG C4-REF",
    "EEG P3-REF",
    "EEG P4-REF",
    "EEG O1-REF",
    "EEG O2-REF",
    "EEG F7-REF",
    "EEG F8-REF",
    "EEG T3-REF",
    "EEG T4-REF",
    "EEG T5-REF",
    "EEG T6-REF",
    "EEG A1-REF",
    "EEG A2-REF",
    "EEG FZ-REF",
    "EEG CZ-REF",
    "EEG PZ-REF",
    "EEG T1-REF",
    "EEG T2-REF",
]

relevant_2channels = ["EEG O1-REF", "EEG T5-REF"]
relevant_3channels = ["EEG O1-REF", "EEG T5-REF", "EEG F7-REF"]
relevant_4channels = ["EEG O1-REF", "EEG T5-REF", "EEG F7-REF", "EEG T4-REF"]
relevant_5channels = ["EEG O1-REF", "EEG T5-REF", "EEG F7-REF", "EEG T4-REF", "EEG CZ-REF"]

standard_channels = [
    "EEG FP1-REF",
    "EEG F7-REF",
    "EEG T3-REF",
    "EEG T5-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F8-REF",
    "EEG T4-REF",
    "EEG T6-REF",
    "EEG O2-REF",
    "EEG FP1-REF",
    "EEG F3-REF",
    "EEG C3-REF",
    "EEG P3-REF",
    "EEG O1-REF",
    "EEG FP2-REF",
    "EEG F4-REF",
    "EEG C4-REF",
    "EEG P4-REF",
    "EEG O2-REF",
]


def split_and_dump(params):

    fetch_folder, sub, dump_folder, label = params
    nb_loaded_files = 0
    for file in os.listdir(fetch_folder):
        if sub in file:
            print("process", file)
            file_path = os.path.join(fetch_folder, file)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            try:
                if drop_channels is not None:
                    useless_chs = []
                    for ch in drop_channels:
                        if ch in raw.ch_names:
                            useless_chs.append(ch)
                    raw.drop_channels(useless_chs)
                for ch in raw.ch_names:
                    if ch not in relevant_3channels:
                        raw.drop_channels(ch)

                print(f"Keeping channels: {raw.ch_names}")

                raw.filter(l_freq=0.1, h_freq=75.0)
                raw.notch_filter(50.0)
                raw.resample(200, n_jobs=5)

                ch_name = raw.ch_names
                raw_data = raw.get_data(units="uV")
                channeled_data = raw_data.copy()
            except:
                with open("tuab-process-error-files.txt", "a") as f:
                    f.write(file + "\n")
                continue

            FS = 200
            WINDOW_SIZE = 300  # 1.5 seconds
            LOW_VAR_LIMIT = 5.0  # Reject windows that are too "quiet"
            HIGH_VAR_LIMIT = 800.0  # Reject massive artifacts (clipping/physical movement)
            SUBSAMPLE_SIZE = 166

            # 1. Skip first 60 seconds (noise)
            start_idx = 60 * FS

            for i in range(start_idx, channeled_data.shape[1] - WINDOW_SIZE, WINDOW_SIZE):
                signal_data = channeled_data[:, i : i + WINDOW_SIZE]  # Shape: (3, 300)

                # Calculate Variance across the time dimension (axis 1)
                # We take the mean variance across all channels
                win_variance = np.mean(np.var(signal_data, axis=1))

                # Decision Logic
                if label == 1:  # For Abnormal files
                    if win_variance < LOW_VAR_LIMIT:
                        continue  # Skip this window; it's likely just normal background

                # Always reject technical artifacts
                if win_variance > HIGH_VAR_LIMIT:
                    continue

                # Subsample and Save
                subsampled_data = sgn.resample(signal_data, SUBSAMPLE_SIZE, axis=1)  # Shape: (3, SUBSAMPLE_SIZE)

                dump_path = os.path.join(dump_folder, f"{file.split('.')[0]}_{i}.pkl")
                pickle.dump({"X": subsampled_data, "y": label}, open(dump_path, "wb"))
                nb_loaded_files += 1
                if nb_loaded_files > 80:
                    print(f"Reached 80 files for subject {sub}, moving to next subject.")
                    break


if __name__ == "__main__":
    """
    TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """
    # root to abnormal dataset
    root = "/path/to/folder/TUAB/edf"
    channel_std = "01_tcp_ar"

    # train, val abnormal subjects
    train_val_abnormal = os.path.join(root, "train", "abnormal", channel_std)
    train_val_a_sub = list(set([item.split("_")[0] for item in os.listdir(train_val_abnormal)]))
    np.random.shuffle(train_val_a_sub)
    train_a_sub = train_val_a_sub

    # train, val normal subjects
    train_val_normal = os.path.join(root, "train", "normal", channel_std)
    train_val_n_sub = list(set([item.split("_")[0] for item in os.listdir(train_val_normal)]))
    np.random.shuffle(train_val_n_sub)
    train_n_sub = train_val_n_sub

    # test abnormal subjects
    test_abnormal = os.path.join(root, "eval", "abnormal", channel_std)
    test_a_sub = list(set([item.split("_")[0] for item in os.listdir(test_abnormal)]))

    # test normal subjects
    test_normal = os.path.join(root, "eval", "normal", channel_std)
    test_n_sub = list(set([item.split("_")[0] for item in os.listdir(test_normal)]))

    # create the train, val, test sample folder
    if not os.path.exists(os.path.join(root, "threechannels")):
        os.makedirs(os.path.join(root, "threechannels"))

    if not os.path.exists(os.path.join(root, "threechannels", "train")):
        os.makedirs(os.path.join(root, "threechannels", "train"))
    train_dump_folder = os.path.join(root, "threechannels", "train")

    if not os.path.exists(os.path.join(root, "threechannels", "val")):
        os.makedirs(os.path.join(root, "threechannels", "val"))
    test_dump_folder = os.path.join(root, "threechannels", "val")

    # fetch_folder, sub, dump_folder, labels
    parameters = []
    counter = 0
    for train_sub in train_a_sub:
        parameters.append([train_val_abnormal, train_sub, train_dump_folder, 1])
        counter += 1
        if counter >= 200:  # Limit to 200 subjects for abnormal class in training
            break
    counter = 0
    for train_sub in train_n_sub:
        parameters.append([train_val_normal, train_sub, train_dump_folder, 0])
        counter += 1
        if counter >= 200:  # Limit to 200 subjects for normal class in training
            break
    counter = 0
    for test_sub in test_a_sub:
        parameters.append([test_abnormal, test_sub, test_dump_folder, 1])
        counter += 1
        if counter >= 50:  # Limit to 50 subjects for abnormal class in testing
            break
    counter = 0
    for test_sub in test_n_sub:
        parameters.append([test_normal, test_sub, test_dump_folder, 0])
        counter += 1
        if counter >= 50:  # Limit to 50 subjects for normal class in testing
            break

    # split and dump in parallel
    with Pool(processes=24) as pool:
        # Use the pool.map function to apply the square function to each element in the numbers list
        result = pool.map(split_and_dump, parameters)

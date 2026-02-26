# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI (2024)
# By Wei-Bang Jiang (2024)
# Based on BIOT code base
# https://github.com/ycq091044/BIOT
# --------------------------------------------------------
import mne
import numpy as np
import pandas as pd
import scipy.signal as sgn
import os
import pickle
from tqdm import tqdm

"""
https://github.com/Abhishaike/EEG_Event_Classification
"""

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

# Map your label IDs to a priority (Higher number = Higher priority)
# Note: Check your specific mapping, but usually:
# SPSW/GPED/PLED are the most important.
priority_map = {
    1: 10,  # GPED (High priority)
    2: 10,  # PLED
    3: 10,  # SPSW
    4: 10,  # TRIP
    5: 5,  # ARTF / EYEM (Medium priority)
    0: 1,  # BCKG (Low priority)
}


def convert_signals(signals, Rawdata):
    signal_names = {k: v for (k, v) in zip(Rawdata.info["ch_names"], list(range(len(Rawdata.info["ch_names"]))))}
    new_signals = np.vstack(
        (
            signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F7-REF"]],  # 0
            (signals[signal_names["EEG F7-REF"]] - signals[signal_names["EEG T3-REF"]]),  # 1
            (signals[signal_names["EEG T3-REF"]] - signals[signal_names["EEG T5-REF"]]),  # 2
            (signals[signal_names["EEG T5-REF"]] - signals[signal_names["EEG O1-REF"]]),  # 3
            (signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F8-REF"]]),  # 4
            (signals[signal_names["EEG F8-REF"]] - signals[signal_names["EEG T4-REF"]]),  # 5
            (signals[signal_names["EEG T4-REF"]] - signals[signal_names["EEG T6-REF"]]),  # 6
            (signals[signal_names["EEG T6-REF"]] - signals[signal_names["EEG O2-REF"]]),  # 7
            (signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F3-REF"]]),  # 14
            (signals[signal_names["EEG F3-REF"]] - signals[signal_names["EEG C3-REF"]]),  # 15
            (signals[signal_names["EEG C3-REF"]] - signals[signal_names["EEG P3-REF"]]),  # 16
            (signals[signal_names["EEG P3-REF"]] - signals[signal_names["EEG O1-REF"]]),  # 17
            (signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F4-REF"]]),  # 18
            (signals[signal_names["EEG F4-REF"]] - signals[signal_names["EEG C4-REF"]]),  # 19
            (signals[signal_names["EEG C4-REF"]] - signals[signal_names["EEG P4-REF"]]),  # 20
            (signals[signal_names["EEG P4-REF"]] - signals[signal_names["EEG O2-REF"]]),  # 21
        )
    )

    keep_channels = [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18, 19, 20, 21]
    return new_signals, keep_channels


def BuildEvents(signals, times, EventData, keep_channels):
    fs = 200.0
    # To get exactly 400 points at 200Hz, we need a 2.0s window
    target_pre_resample_points = 400 
    window_samples = target_pre_resample_points 
    pad_width = int(fs * 2) # Increased padding to be safe for 2s windows

    mask = np.isin(EventData[:, 0], keep_channels)
    df = pd.DataFrame(EventData[mask], columns=["chan", "start", "end", "label_id"])

    # Grouping to keep multi-channel context unique to the time window
    grouped = df.groupby(["start", "end", "label_id"])["chan"].apply(list).reset_index()

    # Pre-pad the signals (16, length)
    signals_padded = np.pad(signals, ((0, 0), (pad_width, pad_width)), mode="edge")

    all_segments = [] 
    final_labels = []

    # Map original channel IDs to 0-15 indices for indexing 'segment'
    chan_map = {orig: i for i, orig in enumerate(keep_channels)}

    for _, row in grouped.iterrows():
        t_start, t_end, label = row["start"], row["end"], row["label_id"]
        chans_present = row["chan"] 

        idx_start = np.searchsorted(times, t_start)
        
        # Centering the 2s window: 0.5s before the 'start' marker
        # This gives you 400 points total
        start_slice = pad_width + idx_start - int(0.5 * fs)
        end_slice = start_slice + window_samples

        # Extract the segment for ALL 16 channels first -> Shape (16, 400)
        segment = signals_padded[:, start_slice:end_slice]
        
        if segment.shape[1] < target_pre_resample_points:
            raise ValueError(f"Segment too short: expected {target_pre_resample_points} points, got {segment.shape[1]} points.")
            # continue # Skip if window goes out of bounds

        # For now, arbitrary chosen channels:
        indices = [chan_map[15], chan_map[19], chan_map[2], chan_map[6], chan_map[3]]  # C3, C4, P3 as an example
        
        reduced_segment = segment[indices, :] 
        resampled_segment = sgn.resample(reduced_segment, 100, axis=1)

        all_segments.append(resampled_segment)
        final_labels.append(label)

    return np.array(all_segments), np.array(final_labels)


def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    if drop_channels is not None:
        useless_chs = []
        for ch in drop_channels:
            if ch in Rawdata.ch_names:
                useless_chs.append(ch)
        Rawdata.drop_channels(useless_chs)
    if chOrder_standard is not None and len(chOrder_standard) == len(Rawdata.ch_names):
        Rawdata.reorder_channels(chOrder_standard)
    if Rawdata.ch_names != chOrder_standard:
        raise ValueError

    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)

    _, times = Rawdata[:]
    signals = Rawdata.get_data(units="uV")
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


def load_up_objects(BaseDir, OutDir):
    for dirName, _, fileList in tqdm(os.walk(BaseDir)):
        print("Found directory: %s" % dirName)
        for fname in fileList:
            if fname[-4:] == ".edf":
                print("\t%s" % fname)
                try:
                    [signals, times, event, Rawdata] = readEDF(
                        dirName + "/" + fname
                    )  # event is the .rec file in the form of an array
                    signals, keep_channels = convert_signals(signals, Rawdata)
                except (ValueError, KeyError):
                    print("something funky happened in " + dirName + "/" + fname)
                    continue
                signals, labels = BuildEvents(signals, times, event, keep_channels)
                for idx, (signal, label) in enumerate(zip(signals, labels)):
                    sample = {
                        "signal": signal,
                        "label": label,
                    }
                    save_pickle(
                        sample,
                        os.path.join(OutDir, fname.split(".")[0] + "-" + str(idx) + ".pkl"),
                    )

    return


def save_pickle(object, filename):
    with open(filename, "wb") as f:
        pickle.dump(object, f)


if __name__ == "__main__":

    """
    TUEV dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """

    root = "/path/to/folder/TUEV/edf"
    train_out_dir = os.path.join(root, "fivechannels", "processed_train")
    eval_out_dir = os.path.join(root, "fivechannels", "processed_eval")
    if not os.path.exists(train_out_dir):
        os.makedirs(train_out_dir)
    if not os.path.exists(eval_out_dir):
        os.makedirs(eval_out_dir)

    BaseDirTrain = os.path.join(root, "train")
    load_up_objects(BaseDirTrain, train_out_dir)

    BaseDirEval = os.path.join(root, "eval")
    load_up_objects(BaseDirEval, eval_out_dir)

    # transfer to train, eval, and test
    root = "/path/to/folder/TUEV/edf/"
    seed = 4523
    np.random.seed(seed)

    train_files_path = os.listdir(os.path.join(root, "fivechannels", "processed_train"))
    test_files_path = os.listdir(os.path.join(root, "fivechannels", "processed_eval"))
    train_sub = list(set([f.split("_")[0] for f in train_files_path]))
    print("train sub", len(train_sub))
    target_train_dir = os.path.join(root, "fivechannels", "train")
    target_test_dir = os.path.join(root, "fivechannels", "val")
    if not os.path.exists(target_train_dir):
        os.makedirs(target_train_dir)
    if not os.path.exists(target_test_dir):
        os.makedirs(target_test_dir)

    val_sub = []
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files_path if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files_path if f.split("_")[0] in train_sub]

    for file in train_files:
        os.system(
            f"mv {os.path.join(root, 'fivechannels', 'processed_train', file)} {target_train_dir}"
        )
    for file in test_files_path:
        os.system(
            f"mv {os.path.join(root, 'fivechannels', 'processed_eval', file)} {target_test_dir}"
        )

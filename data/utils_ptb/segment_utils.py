import scipy
import numpy as np
import torch
import pandas as pd


def detect_rpeaks(ecg, rate=200, ransac_window_size=5.0, lowfreq=5.0, highfreq=15.0):
    import warnings

    warnings.filterwarnings("ignore")

    # Improved Bandpass Filter (5-15 Hz is the QRS sweet spot)
    nyq = rate / 2.0
    b, a = scipy.signal.butter(2, [lowfreq / nyq, highfreq / nyq], btype="band")
    ecg_band = scipy.signal.filtfilt(b, a, ecg)

    # Derivative + Squaring (Highlight the steep slopes of the R-wave)
    decg = np.diff(ecg_band)
    decg_power = decg**2

    # Robust Thresholding (RANSAC-like)
    ransac_samples = int(ransac_window_size * rate)
    thresholds = []
    for i in range(max(1, int(len(decg_power) / ransac_samples))):
        d = decg_power[i * ransac_samples : (i + 1) * ransac_samples]
        if len(d) > 0:
            thresholds.append(0.5 * np.std(d))

    threshold = np.median(thresholds) if thresholds else 0
    decg_power[decg_power < threshold] = 0

    # Shannon Energy (Enveloping)
    decg_power = decg_power / (np.max(decg_power) + 1e-10)
    shannon_energy = -(decg_power**2) * np.log(decg_power**2 + 1e-10)

    # Integration (Smoothing)
    # 0.125s window is standard for heartbeat integration
    mean_window_len = int(rate * 0.125 + 1)
    lp_energy = np.convolve(shannon_energy, np.ones(mean_window_len) / mean_window_len, mode="same")
    lp_energy = scipy.ndimage.gaussian_filter1d(lp_energy, rate / 10.0)

    # Peak Finding via Zero Crossings of derivative
    lp_energy_diff = np.diff(lp_energy)
    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    peaks = np.flatnonzero(zero_crossings)

    return peaks


def find_rpeaks_clean_ecgs_in_dataframe(data: pd.DataFrame, ref_channel_idx: int = 1, rate: int = 500) -> pd.DataFrame:

    def find_rpeaks_multi(signal_2d):
        # signal_2d shape: (5000, 12) or (5000, num_chosen)
        # We detect peaks based ONLY on the reference channel
        reference_signal = signal_2d[:, ref_channel_idx]
        return detect_rpeaks(ecg=reference_signal, rate=rate)

    df_clean = data.copy()
    # Apply detection to the multi-channel signal
    df_clean["rpeaks_indexes"] = df_clean["ecg_signal_raw"].apply(find_rpeaks_multi)

    # FILTER: Retrieve only patients who have at least one peak
    df_clean = df_clean[df_clean["rpeaks_indexes"].apply(len) >= 1]
    # if save: df_clean.to_pickle(filename + '_clean_with_rpeaks_indexes.pkl')
    return df_clean


def segment_ecg_in_clean_dataframe(
    ROOT: str = ".",
    data: pd.DataFrame = None,
    size_before_index: int = 200,
    size_after_index: int = 300,
    signal_length: int = 5000,
) -> pd.DataFrame:

    def get_heartbeats_indexes(
        indexes, size_before_index=size_before_index, size_after_index=size_after_index, signal_length=signal_length
    ):
        if len(indexes) == 0:
            return []
        # Filter peaks that are too close to the start or end to extract a full window
        valid_peaks = [x for x in indexes if (signal_length - size_after_index) > x > size_before_index]
        # Create [start, end] pairs
        return [[p - size_before_index, p + size_after_index] for p in valid_peaks]

    def split_ecgs_multichannel(ecg, indexes_pairs):
        """
        ecg: numpy array of shape (5000, num_channels)
        indexes_pairs: list of [start, end] pairs
        """
        if not indexes_pairs:
            return []

        heart_beats = []
        for start, end in indexes_pairs:
            # Slicing a 2D array [start:end, :] keeps all channels for that window
            segment = ecg[start:end, :]  # or try ecg[start:end] if ecg is already in shape (5000, num_channels)
            heart_beats.append(segment)

        # Stack into shape: (num_heartbeats, window_length, num_channels)
        return np.array(heart_beats)

    df = data.copy()

    df["heartbeat_indexes"] = df["rpeaks_indexes"].apply(lambda x: get_heartbeats_indexes(x))

    # Extract segments (Maintains multichannel structure)
    df["ecg_signal_heartbeat"] = df.apply(
        lambda x: split_ecgs_multichannel(x["ecg_signal_raw"], x["heartbeat_indexes"]), axis=1
    )

    return df


def get_heartbeats_indexes(indexes, size_before_index=200, size_after_index=300):
    if len(indexes) == 0:
        return []
    indexes_ = [x for x in indexes if 5000 - size_after_index >= x >= size_before_index]
    indexes_new = [[index - size_before_index] + [index + size_after_index] for index in indexes_]
    return indexes_new


def split_ecgs(ecg, indexes, size_before_index=200, size_after_index=300):
    import itertools

    if len(indexes) == 0:
        return []
    indexes_final = list(itertools.chain(*get_heartbeats_indexes(indexes, size_before_index, size_after_index)))
    heart_beats = np.split(ecg, indexes_final)[1::2]  # [:-1]
    arr = np.stack(heart_beats, axis=0) if len(heart_beats) > 1 else heart_beats
    return arr


def save_x_y_numpy(uuid: str, df_all: pd.DataFrame, partition: str = "training"):
    noisy_hb = df_all[
        (df_all["partition"] == partition) & (df_all["ecg_signal_noised"] == True) & (df_all["ecg_origin_uuid"] == uuid)
    ].ecg_signal_heartbeat.values[:]
    if noisy_hb.shape[0] < 2:
        return None, None
    arr_noisy = np.vstack(noisy_hb)
    clean_hb = df_all[
        (df_all["partition"] == partition)
        & (df_all["ecg_signal_noised"] == False)
        & (df_all["ecg_origin_uuid"] == uuid)
    ].ecg_signal_heartbeat.values[:][0]
    arr_clean = np.repeat(clean_hb, repeats=arr_noisy.shape[0] // clean_hb.shape[0], axis=0)
    return arr_clean, arr_noisy


def values_from_dataframe_ny_list(df: pd.DataFrame, key: str, as_list=False):
    val = df[key].values[:]
    values_arr = np.zeros([len(val)] + [s for s in val[0].shape]) if not as_list else []
    len_arr = []

    for i in range(len(val)):
        if not as_list:
            values_arr[i, :] = val[i]
        else:
            len_arr.append(len(val[i]))
            values_arr.append(val[i])
    return values_arr, len_arr

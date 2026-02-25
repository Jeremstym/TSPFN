import pandas as pd
import csv
import gzip
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

FILTERED_FOLDER = "/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/filtered_multi_channel_ts"
IMPUTED_FOLDER = "/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/imputed_multi_channel_ts"
IMPUTED_FOLDER_decease = "/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/imputed_multi_channel_ts_decease"
SPLIT_TRAIN = "/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/train_decease"
SPLIT_TEST = "/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/test_decease"


def extract_multi_channel_vitals(min_valid_points=100):
    """
    Processes vitalPeriodic once, aligns 5 channels, and skips low-quality stays.
    min_valid_points: Minimum non-NaN values required PER CHANNEL to save the patient.
    """
    PATH_VITAL_PERIODIC = "/data/stympopper/BenchmarkTSPFN/EICU-CRD/vitalPeriodic.csv.gz"
    DESTINATION_FOLDER = FILTERED_FOLDER

    CHANNELS = {"heartrate": 0, "respiration": 1, "sao2": 2, "systemicmean": 3, "temperature": 4}
    CHANNEL_NAMES = ["heart_rate", "respiration", "spo2", "blood_pressure", "temperature"]

    # if not os.listdir(DESTINATION_FOLDER): # Only create if empty/missing
    os.makedirs(DESTINATION_FOLDER, exist_ok=True)

    def save_patient_bundle(pid, records):
        if not pid or not records:
            return

        records.sort(key=lambda x: x["offset"])
        offsets = np.array([r["offset"] for r in records], dtype=np.float32)
        values = np.full((len(records), len(CHANNELS)), np.nan, dtype=np.float32)

        for i, rec in enumerate(records):
            for col_name, ch_idx in CHANNELS.items():
                values[i, ch_idx] = rec[col_name]

        # --- QUALITY CONTROL CHECK ---
        # Count non-NaN values for each channel
        valid_counts = np.count_nonzero(~np.isnan(values), axis=0)

        # If ANY channel has fewer than min_valid_points, we skip this patient
        if np.any(valid_counts < min_valid_points):
            # Optimization: could log skipped IDs to a file if needed
            return

        file_path = os.path.join(DESTINATION_FOLDER, f"{pid}.npz")
        np.savez_compressed(file_path, data=values, offsets=offsets, columns=CHANNEL_NAMES)

    print(f"Starting Extraction (Threshold: {min_valid_points} points per channel)...")

    with gzip.open(PATH_VITAL_PERIODIC, "rt") as f:
        reader = csv.DictReader(f)
        current_pid = None
        current_records = []

        # Use a status bar that updates based on line count if possible,
        # or just a periodic print
        for row in reader:
            pid = row["patientunitstayid"]
            try:
                record = {
                    "offset": float(row["observationoffset"]),
                    "heartrate": float(row["heartrate"]) if row["heartrate"] else np.nan,
                    "respiration": float(row["respiration"]) if row["respiration"] else np.nan,
                    "sao2": float(row["sao2"]) if row["sao2"] else np.nan,
                    "systemicmean": float(row["systemicmean"]) if row["systemicmean"] else np.nan,
                    "temperature": float(row["temperature"]) if row["temperature"] else np.nan,
                }
            except ValueError:
                continue

            if pid != current_pid:
                if current_pid is not None:
                    save_patient_bundle(current_pid, current_records)
                current_pid = pid
                current_records = [record]
            else:
                current_records.append(record)

        save_patient_bundle(current_pid, current_records)

    print(f"Done! Multi-channel files saved in {DESTINATION_FOLDER}")


def get_imputed_block(ts, window_size=100, max_nan_ratio=0.3):
    """
    Finds a window of 100 rows where NaNs are below a threshold,
    then imputes the values.
    """
    # 1. Calculate how many NaNs are in every possible window
    # We sum the NaNs across all 5 channels for each row
    nans_per_row = np.isnan(ts).any(axis=1).astype(int)

    if len(ts) < window_size:
        raise ValueError("Time series is shorter than the window size.")

    # Sliding window sum of NaN-containing rows
    kernel = np.ones(window_size)
    nan_counts = np.convolve(nans_per_row, kernel, mode="valid")

    # 2. Find windows that meet our tolerance (e.g., < 30% rows have NaNs)
    valid_starts = np.where(nan_counts <= (window_size * max_nan_ratio))[0]

    if valid_starts.size == 0:
        return None

    # Take the window with the absolute minimum number of NaNs
    best_start = valid_starts[np.argmin(nan_counts[valid_starts])]
    block = ts[best_start : best_start + window_size].copy()

    # 3. Impute using Pandas (the most efficient way for FFilling)
    df_block = pd.DataFrame(block)

    # Step-wise imputation:
    df_block = df_block.interpolate(method="linear", limit_direction="both")  # Interpolate
    df_block = df_block.ffill()  # Forward fill
    df_block = df_block.bfill()  # Backward fill (for gaps at the very start)

    return df_block.values


def filter_imputed_blocks(source_folder, target_folder, window_size=100, max_nan_ratio=0.3, median=None):
    os.makedirs(target_folder, exist_ok=True)
    for file in tqdm(os.listdir(source_folder)):
        if not file.endswith(".npz"):
            raise ValueError(f"Unexpected file format: {file}. Expected .npz files.")
        data = np.load(os.path.join(source_folder, file))["data"]
        # imputed_block = get_imputed_block(data, window_size, max_nan_ratio)
        # imputed_block = get_terminal_100_points(data, window_size, median=median)
        # imputed_block = get_window_with_gap(data, window_size=window_size, gap_size=48, median=median)
        imputed_block = get_window_with_gap_v2(data, window_size=window_size, gap_size=48, global_medians=median)
        if imputed_block is not None:
            np.savez_compressed(os.path.join(target_folder, file), data=imputed_block)


def create_final_labels(ts_folder, patient_csv_path, output_path):
    """
    Links extracted time series files to their mortality status.
    """
    print("Generating Master Label CSV...")

    # 1. Load the ground truth from the patient table
    # hospitaldischargestatus: 'Expired' = 1, anything else = 0
    df_patient = pd.read_csv(
        patient_csv_path, usecols=["patientunitstayid", "hospitaldischargestatus"], compression="gzip"
    )
    df_patient["mortality_label"] = (df_patient["hospitaldischargestatus"] == "Expired").astype(int)

    # Create a mapping for quick lookup
    label_map = dict(zip(df_patient["patientunitstayid"], df_patient["mortality_label"]))

    # 2. Get the list of patients who actually have saved time series data
    # We only want to label patients we can actually train on
    extracted_pids = [int(f.split(".")[0]) for f in os.listdir(ts_folder) if f.endswith(".npz")]

    final_data = []
    for pid in tqdm(extracted_pids, desc="Linking IDs to labels"):
        if pid in label_map:
            final_data.append({"patientunitstayid": pid, "mortality_label": label_map[pid]})

    # 3. Save to CSV
    df_labels = pd.DataFrame(final_data)
    df_labels.to_csv(output_path, index=False)

    print(f"Success! Created labels for {len(df_labels)} patients.")
    print(f"Mortality Rate in this subset: {round(df_labels['mortality_label'].mean() * 100, 2)}%")


def split_train_val(label_csv, train_ratio=0.8):
    df = pd.read_csv(label_csv)

    df_train, df_val = train_test_split(df, test_size=1 - train_ratio, random_state=42, stratify=df["mortality_label"])
    target_dir_train = SPLIT_TRAIN
    target_dir_val = SPLIT_TEST
    os.makedirs(target_dir_train, exist_ok=True)
    os.makedirs(target_dir_val, exist_ok=True)

    for _, row in df_train.iterrows():
        pid = row["patientunitstayid"]
        src_file = os.path.join(IMPUTED_FOLDER_decease, f"{pid}.npz")
        dst_file = os.path.join(target_dir_train, f"{pid}.npz")
        if os.path.exists(src_file):
            os.rename(src_file, dst_file)
        else:
            print(f"Warning: Source file {src_file} not found for training set.")

    for _, row in df_val.iterrows():
        pid = row["patientunitstayid"]
        src_file = os.path.join(IMPUTED_FOLDER_decease, f"{pid}.npz")
        dst_file = os.path.join(target_dir_val, f"{pid}.npz")
        if os.path.exists(src_file):
            os.rename(src_file, dst_file)
        else:
            print(f"Warning: Source file {src_file} not found for validation set.")


def calculate_true_medians(source_dir=FILTERED_FOLDER):
    all_values = []
    patient_files = [f for f in os.listdir(source_dir) if f.endswith(".npz")]

    # We only need a sample to get a stable median
    for filename in patient_files[:2000]:
        data = np.load(os.path.join(source_dir, filename))["data"]
        all_values.append(data)

    # Stack everything and calculate median per column, ignoring NaNs
    big_matrix = np.vstack(all_values)
    medians = np.nanmedian(big_matrix, axis=0)

    print(f"Calculated Medians [HR, Resp, SpO2, BP, Temp]: {medians}")
    return medians


def get_terminal_100_points(ts, window_size=100, median: list = None):
    """
    ts: array of shape (N, 5)
    Always returns (100, 5)
    """
    N = ts.shape[0]

    if N >= window_size:
        # 1. Long Stay: Take the most recent 100 points
        block = ts[-window_size:].copy()
    else:
        # 2. Short Stay: Pad the beginning with NaNs
        pad_width = window_size - N
        block = np.pad(ts, ((pad_width, 0), (0, 0)), mode="constant", constant_values=np.nan)

    # 3. Impute Gaps
    # Convert to DataFrame to use clinical imputation logic
    df = pd.DataFrame(block)

    # Linear interpolation for small gaps, Forward fill for the rest
    # Backward fill handles the Padding we just added at the start
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()

    # 4. Global Median Fallback (for patients missing a whole channel)
    # [HR, Resp, SpO2, MAP, Temp]
    # global_medians = [85, 18, 97, 80, 37]
    global_medians = calculate_true_medians() if median is None else median
    if df.isnull().values.any():
        for i, median_val in enumerate(global_medians):
            df.iloc[:, i] = df.iloc[:, i].fillna(median_val)

    return df.values


def get_window_with_gap(ts, window_size=100, gap_size=48, median: list = None):
    """
    ts: array of shape (N, 5)
    Always returns (100, 5) or None if stay is too short for any data
    """
    N = ts.shape[0]

    # 1. Define the 'End' of our observation (4 hours before discharge)
    end_idx = N - gap_size

    # 2. If the patient died/left before the gap even started, we have no valid data
    if end_idx <= 0:
        return None

    # 3. Define the 'Start' of our observation
    start_idx = end_idx - window_size

    if start_idx >= 0:
        # CASE: Long Stay
        # We have enough data to take a full 100-point block ending at the gap
        block = ts[start_idx:end_idx].copy()
    else:
        # CASE: Short Stay
        # We take everything from admission (0) up to the gap (end_idx)
        available_data = ts[0:end_idx].copy()

        # We need to pad the beginning to reach exactly 100
        pad_width = window_size - len(available_data)
        # Pre-pad with NaNs (which we will impute/median-fill later)
        block = np.pad(available_data, ((pad_width, 0), (0, 0)), mode="constant", constant_values=np.nan)

        # block = pd.DataFrame(block)
        # block = block.interpolate(method='linear', limit_direction='both').ffill().bfill()
        global_medians = calculate_true_medians() if median is None else median
        if np.isnan(block).any():
            for i, median_val in enumerate(global_medians):
                block[:, i] = np.where(np.isnan(block[:, i]), median_val, block[:, i])

    return block


def get_window_with_gap_v2(ts, window_size=100, gap_size=48, global_medians=None):
    N = ts.shape[0]
    end_idx = N - gap_size
    if end_idx <= 0:
        return None

    start_idx = max(0, end_idx - window_size)
    block = ts[start_idx:end_idx].copy()

    # Create a mask: 1 for real data, 0 for missing/padded
    mask = (~np.isnan(block)).astype(np.float32)

    # 1. Padding if too short
    if len(block) < window_size:
        pad_len = window_size - len(block)
        block = np.pad(block, ((pad_len, 0), (0, 0)), constant_values=np.nan)
        mask = np.pad(mask, ((pad_len, 0), (0, 0)), constant_values=0)

    # 2. Advanced Imputation using Pandas
    df = pd.DataFrame(block)
    # Fill small gaps with linear lines, then carry values forward
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()

    # 3. Last Resort: Global Medians (if a channel is 100% missing)
    if global_medians is not None:
        df = df.fillna(pd.Series(global_medians))

    final_block = df.values.astype(np.float32)

    # Optional: Concatenate mask to create a (100, 10) input
    # This is HIGHLY recommended for SOTA performance
    # final_block = np.concatenate([final_block, mask], axis=1)

    return final_block


if __name__ == "__main__":
    # extract_multi_channel_vitals()
    median = calculate_true_medians()
    filter_imputed_blocks(
        source_folder=FILTERED_FOLDER,
        target_folder=IMPUTED_FOLDER_decease,
        window_size=100,
        max_nan_ratio=0.3,
        median=median,
    )
    create_final_labels(
        ts_folder=IMPUTED_FOLDER_decease,
        patient_csv_path="/data/stympopper/BenchmarkTSPFN/EICU-CRD/patient.csv.gz",
        output_path="/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/final_labels.csv",
    )
    split_train_val(
        label_csv="/data/stympopper/BenchmarkTSPFN/processed/EICU_preprocessed/final_labels.csv", train_ratio=0.8
    )

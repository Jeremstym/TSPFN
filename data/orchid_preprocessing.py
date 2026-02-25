import os
import sys
import numpy as np
import scipy.signal as sgn
from scipy.interpolate import CubicSpline, PchipInterpolator
import pandas as pd
from glob import glob
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split

origin_path = "/data/stympopper/ORCHID/database"
target_dir = "/data/stympopper/BenchmarkTSPFN/processed/ORCHID_processed/data"
label_dir = "/data/stympopper/BenchmarkTSPFN/processed/ORCHID_processed"

label_list = ["amyloidosis", "hta", "cmi", "CMD", "healthy"]
label_map = {label: idx for idx, label in enumerate(label_list)}

label_dict = {}

def process_gls(strain_data, target_len=64):
    data_consistent = np.copy(strain_data)
    # data_consistent[-1] = data_consistent[0]
    # Create the original time index (0 to 1)
    x = np.linspace(0, 1, len(data_consistent))
    # Create the new 64-point time index
    x_new = np.linspace(0, 1, target_len)
    
    # Use 'periodic' bc_type if your cycle starts and ends at End-Diastole
    # This ensures a smooth transition if you were to loop the signal
    cs = CubicSpline(x, data_consistent, bc_type='natural')
    
    return cs(x_new)

def clean_and_resample(data, target_len=64):
    # Basic quality check: if signal is too short to be a cycle, return None
    # if len(data) < 20: 
    #     return None 
    
    x_old = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, target_len)
    
    # PCHIP is excellent for biological 'valleys' and 'peaks'
    interp_func = PchipInterpolator(x_old, data)
    resampled_signal = interp_func(x_new)
    
    return resampled_signal

def resample_signals(path, target_dir, label_dir, target_sample=64):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    list_A4C = glob(path + "/[0-9]*/[0-9]*_A4C_mask.npz")
    for file in tqdm(list_A4C):
        data = np.load(file)
        valid_patient = True
    
        # # First pass: Check if ALL required features meet the length requirement
        # for feature in data.files:
        #     if len(data[feature]) < 20:
        #         print(f"Skipping FULL PATIENT {file}: {feature} too short ({len(data[feature])})")
        #         valid_patient = False
        #         break
                
        if not valid_patient:
            continue
            
        patient_id = file.split("/")[-1].split("_")[0]
        with open(file.replace("_A4C_mask.npz", ".yaml"), "r") as f:
            yaml_data = yaml.safe_load(f)
        label = label_map[yaml_data["diagnosis"]]
        label_dict[patient_id] = label

        patient_ts = {}
        for feature in data.files:
            signal = data[feature]
            resampled_signal = sgn.resample(signal, target_sample)
            # resampled_signal = process_gls(signal, target_len=target_sample)
            # resampled_signal = clean_and_resample(signal, target_len=target_sample)
            patient_ts[feature] = resampled_signal
        np.savez(os.path.join(target_dir, f"{patient_id}_A4C_mask.npz"), **patient_ts)

    df_label = (
        pd.DataFrame.from_dict(label_dict, orient="index", columns=["diagnosis"])
        .reset_index()
        .rename(columns={"index": "patient_id"})
    )
    df_label.to_csv(os.path.join(label_dir, "labels.csv"), index=False)

    print(f"Resampled signals saved to {target_dir}")


def split_train_val(path, label_csv, train_ratio=0.8):
    df = pd.read_csv(label_csv)

    df_train, df_val = train_test_split(df, test_size=1 - train_ratio, random_state=42, stratify=df["diagnosis"])
    target_dir_train = os.path.join(path, "train")
    target_dir_val = os.path.join(path, "val")
    os.makedirs(target_dir_train, exist_ok=True)
    os.makedirs(target_dir_val, exist_ok=True)

    for _, row in df_train.iterrows():
        patient_id = row["patient_id"]
        src_file = os.path.join(path, f"{patient_id}_A4C_mask.npz")
        dst_file = os.path.join(target_dir_train, f"{patient_id}_A4C_mask.npz")
        os.rename(src_file, dst_file)
    for _, row in df_val.iterrows():
        patient_id = row["patient_id"]
        src_file = os.path.join(path, f"{patient_id}_A4C_mask.npz")
        dst_file = os.path.join(target_dir_val, f"{patient_id}_A4C_mask.npz")
        os.rename(src_file, dst_file)

    print(f"Data split into train and val sets. Train samples: {len(df_train)}, Val samples: {len(df_val)}")

if __name__ == "__main__":
    resample_signals(origin_path, target_dir, label_dir, target_sample=64)
    split_train_val(target_dir, os.path.join(label_dir, "labels.csv"))

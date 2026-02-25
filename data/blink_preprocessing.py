import os
import csv
import gzip
import numpy as np
import pandas as pd
from sktime.datasets import load_from_tsfile
from sktime.datatypes import convert_to
from tqdm import tqdm
from sklearn.model_selection import train_test_split

ORIGIN_DIRECTORY = "/path/to/folder/Blink/"
OUTPUT_DIRECTORY = "/path/to/folder/processed/Blink/"
SPLIT_TRAIN = "train"
SPLIT_TEST = "test"

def process_blink_data(origin_dir, output_dir, split_train, split_test):
    X_train, y_train = load_from_tsfile(os.path.join(origin_dir, "Blink_TRAIN.ts"))
    X_test, y_test = load_from_tsfile(os.path.join(origin_dir, "Blink_TEST.ts"))

    X_train_numpy = convert_to(X_train, "numpy3D")
    X_test_numpy = convert_to(X_test, "numpy3D")

    label_map = {"longblink": 0.0, "shortblink": 1.0}
    y_train_mapped = np.array([label_map[label] for label in y_train])
    y_test_mapped = np.array([label_map[label] for label in y_test])

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{split_train}_features.npy"), X_train_numpy)
    np.save(os.path.join(output_dir, f"{split_train}_labels.npy"), y_train_mapped)
    np.save(os.path.join(output_dir, f"{split_test}_features.npy"), X_test_numpy)
    np.save(os.path.join(output_dir, f"{split_test}_labels.npy"), y_test_mapped)

    return

if __name__ == "__main__":
    process_blink_data(
        origin_dir=ORIGIN_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        split_train=SPLIT_TRAIN,
        split_test=SPLIT_TEST,
    )
import torch
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, TensorDataset


from utils.DataPreparation.data_preparation import Data_Preparation
from utils.DataPreparation.data_deepfade import load_features, load_dataframe_clean_ecg_II, paths
from utils.segment_utils import undersample_signal

def get_features_labels(args):
    data_name = args.data_name
    print(data_name)
    assert data_name in ['deepfade', 'descod', 'tdp']

    if data_name == 'deepfade':
        X_train, y_train, X_val, y_val, X_test, y_test = get_data_deepfade(return_as_numpy=True, signal_size=args.s_size, data_source=args.data_source, train=args.is_train)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test = get_data_competitors(args, return_as_numpy=True)

    return X_train, y_train, X_val, y_val, X_test, y_test

def get_data_competitors(n_type, return_as_numpy: bool=False):
    [X_train, y_train, X_test, y_test] = Data_Preparation(n_type)

    X_train = torch.FloatTensor(X_train)
    X_train = X_train.permute(0, 2, 1)

    y_train = torch.FloatTensor(y_train)
    y_train = y_train.permute(0, 2, 1)

    # print(X_train.shape, y_train.shape)

    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0, 2, 1)

    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0, 2, 1)

    train_val_set = TensorDataset(y_train, X_train)
    test_set = TensorDataset(y_test, X_test)

    train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.3)
    train_set = Subset(train_val_set, train_idx)
    val_set = Subset(train_val_set, val_idx)

    if return_as_numpy:
        return X_train[train_idx, :], y_train[train_idx, :], X_train[val_idx, :], y_train[val_idx, :], X_test, y_test

    return train_set, val_set, test_set

def get_data_deepfade(return_as_numpy: bool=False, signal_size: int=500, train: bool=True, data_source: str='cache'):
    """
    This function returns the numpy arrays of the deepfade data: Generapol + ECGMMLD + ECGRVDQ + NSTDB of Lead II
    @data_source: 'cache' to extract the 5000 points signal, ''heartbeats' otherwise
    """

    global y_test
    for i in tqdm(range(10), ascii=True, desc='Loop', ncols=0):
        if i == 0:
            X_train = load_features(batch_id=i, partition='train', type='signals', data_source=data_source).unsqueeze(1)
            y_train = load_features(batch_id=i, partition='train', type='labels', data_source=data_source).unsqueeze(1)
            if i < 9:
                X_val = load_features(batch_id=i, partition='val', type='signals', data_source=data_source).unsqueeze(1)
                y_val = load_features(batch_id=i, partition='val', type='labels', data_source=data_source).unsqueeze(1)
                X_test = load_features(batch_id=i, partition='holdout', type='signals', data_source=data_source).unsqueeze(1)
                y_test = load_features(batch_id=i, partition='holdout', type='labels', data_source=data_source).unsqueeze(1)

        else:
            X_train = torch.vstack((X_train, load_features(batch_id=i, partition='train', type='signals', data_source=data_source).unsqueeze(1)))
            y_train = torch.vstack((y_train, load_features(batch_id=i, partition='train', type='labels', data_source=data_source).unsqueeze(1)))
            if i < 9:
                X_val = torch.vstack((X_val, load_features(batch_id=i, partition='val', type='signals', data_source=data_source).unsqueeze(1)))
                y_val = torch.vstack((y_val, load_features(batch_id=i, partition='val', type='labels', data_source=data_source).unsqueeze(1)))
                X_test = torch.vstack((X_test, load_features(batch_id=i, partition='holdout', type='signals', data_source=data_source).unsqueeze(1)))
                y_test = torch.vstack((y_test, load_features(batch_id=i, partition='holdout', type='labels', data_source=data_source).unsqueeze(1)))

    if signal_size == 512:
        # print(f"Converting the signal to size {signal_size}")

        if train:

            X_train = torch.Tensor(undersample_signal(X_train, signal_size))
            y_train = torch.Tensor(undersample_signal(y_train, signal_size))
            # print("Shape X_train, y_train:", X_train.shape, y_train.shape)

            X_val = torch.Tensor(undersample_signal(X_val, signal_size))
            y_val = torch.Tensor(undersample_signal(y_val, signal_size))
            # print("Shape X_val, y_val:", X_val.shape, y_val.shape)

        else:
            X_test = torch.Tensor(undersample_signal(X_test, signal_size))
            y_test = torch.Tensor(undersample_signal(y_test, signal_size))
            # print("Shape X_test, y_test:", X_test.shape, y_test.shape)

    if return_as_numpy:

        return X_train, y_train, X_val, y_val, X_test, y_test


    else:
        train_set = TensorDataset(X_train, y_train)
        val_set = TensorDataset(X_val, y_val)
        test_set = TensorDataset(X_test, y_test)

        return train_set, val_set, test_set

def load_clean_ecg_ny(dataset_name: str='Generepol', partition: str='all') -> np.ndarray:
    """
    Loads the dataframe containing the clean ECG of lead II from dataset_name and returns the numpy array associated with the partition.
    ONLY LEAD II
    """
    df = load_dataframe_clean_ecg_II(dataset_name, partition)
    signals = df.ecg_signal_raw.values[:]
    signals_arr = np.zeros([len(signals), signals[0].shape[0]])
    for i in range(len(signals)):
        signals_arr[i, :] = signals[i]
    return signals_arr

def load_heartbeats_ecg_generepol(dataset_name: str='Generepol', partition: str='all') -> np.ndarray:
    """
    Load the numpy array containing the heartbeats in 5000 points. Available only for lead II, dataset Generepol and partition holdout
    and training.
    """
    assert dataset_name in ['Generepol'], "Invalid dataset, possible 'Generepol'"
    assert partition in ['holdout', 'training'], "Invalid dataset, possible 'Generepol'"
    root = paths['data_heartbeats'] + '/occluded_5000/'
    arr_list = []
    for batch_id in tqdm(range(9), ncols=0):
        arr = np.load(f"{root}batch_{batch_id}_hb_lead_II_{partition}_{dataset_name}.npy")
        arr_list.append(arr)
    return np.vstack(arr_list)


def load_array_from_name(denoiser_name, noise_signal='BW'):
    if denoiser_name == 'original':
        signal = np.load(f'experiment_data/denoising_task/hb_holdout.npy').squeeze()
    else:
        signal = np.load(f'experiment_data/denoising_task/hb_holdout_{noise_signal}{"_" + denoiser_name if denoiser_name != "noise" else ""}.npy').squeeze()
    if signal.shape[1] != 500:
        import neurokit2 as nk
        signal = np.apply_along_axis(nk.signal_resample, axis=1, arr=signal, method="FFT", desired_length=500)
    return signal


def filter_noise_data(data,
                      threshold=2):
    def has_consecutive_constant(row,
                                 threshold):
        consecutive_count = 1
        max_consecutive_count = 1
        for i in range(1, len(row)):
            if row[i] == row[i - 1]:
                consecutive_count += 1
                max_consecutive_count = max(max_consecutive_count, consecutive_count)
            else:
                consecutive_count = 1
        return max_consecutive_count > threshold  # len(row) / 2

    mask = np.array([not has_consecutive_constant(row, threshold) for row in data])
    return mask

def normalize_signals_for_comparison(signal_1_d):
    return (signal_1_d - signal_1_d.min()) / (signal_1_d.max() - signal_1_d.min())

if __name__ == '__main__':
    ecgs = load_heartbeats_ecg_generepol('Generepol', 'holdout')
    # print(ecgs.shape)

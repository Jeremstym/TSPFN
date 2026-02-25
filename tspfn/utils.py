import numpy as np
import os
import faiss

# from faiss import StandardGpuResources
import random
import torch
from torch import Tensor
from typing import Tuple, Union, Dict, Sequence
import torch.nn as nn
import shutil
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def stratified_batch_split(data: Tensor, labels: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    unique_labels = torch.unique(labels)
    support_indices = []
    query_indices = []

    for label in unique_labels:
        # Find all indices for this specific class
        idx = (labels == label).nonzero(as_tuple=True)[0]

        # Split these indices in half
        mid = len(idx) // 2
        support_indices.append(idx[:mid])
        query_indices.append(idx[mid:])

    # Concatenate indices
    support_indices = torch.cat(support_indices)
    query_indices = torch.cat(query_indices)

    return data[support_indices], data[query_indices], labels[support_indices], labels[query_indices]


def half_batch_split(data: Tensor, labels: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    if data.ndim == 2:
        total_size = data.size(0)
        half_size = total_size // 2

        all_indices = list(range(total_size))

        support_indices = all_indices[:half_size]
        query_indices = all_indices[half_size:]
        support_data = data[support_indices]
        query_data = data[query_indices]
        support_labels = labels[support_indices]
        query_labels = labels[query_indices]

    elif data.ndim == 3:
        # data: [Batch, Samples, Features]
        # labels: [Batch, Samples]

        total_samples = data.size(1)
        half_size = total_samples // 2

        # Split Data on dimension 1
        support_data = data[:, :half_size, :]
        query_data = data[:, half_size:, :]

        # Split Labels on dimension 1
        support_labels = labels[:, :half_size]
        query_labels = labels[:, half_size:]

    else:
        raise ValueError("Data tensor must be 2D or 3D")

    return support_data, query_data, support_labels, query_labels


def z_scoring(
    data_support: Tensor, data_query: Tensor, label_support: Tensor, label_query: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    if data_support.ndim == 3:
        # data: [Batch, Samples, Features]
        # We compute mean and std on the sample dimension (dim=1)
        mean = data_support.mean(dim=1, keepdim=True)
        std = data_support.std(dim=1, keepdim=True) + 1e-8  # Avoid division by zero

        data_support_z = (data_support - mean) / std
        data_query_z = (data_query - mean) / std

    elif data_support.ndim == 2:
        # data: [Batch, Features]
        mean = data_support.mean(dim=0, keepdim=True)
        std = data_support.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero

        data_support_z = (data_support - mean) / std
        data_query_z = (data_query - mean) / std

    else:
        raise ValueError("Data tensor must be 2D or 3D")

    return data_support_z, data_query_z, label_support, label_query


def z_scoring_per_channel(
    data_support: Tensor, data_query: Tensor, label_support: Tensor, label_query: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    if data_support.ndim == 4:
        # data: [Batch=1, Samples, Channels, Time]
        # We compute mean and std on the sample dimension (dim=1)
        mean = data_support.mean(dim=1, keepdim=True)
        std = data_support.std(dim=1, keepdim=True) + 1e-8  # Avoid division by zero

        data_support_z = (data_support - mean) / std
        data_query_z = (data_query - mean) / std

    elif data_support.ndim == 3:
        # data: [Samples, Channels, Time]
        mean = data_support.mean(dim=0, keepdim=True)
        std = data_support.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero

        data_support_z = (data_support - mean) / std
        data_query_z = (data_query - mean) / std

    else:
        raise ValueError("Data tensor must be 4D or 3D (with Batch=1)")

    return data_support_z, data_query_z, label_support, label_query


def get_stratified_batch_split(data, labels, n_total=10000):
    device = data.device

    min_support = 16
    max_support = n_total - int(0.5 * n_total)  # We keep at least 50% for the query to have a meaningful evaluation

    log_min, log_max = np.log(min_support), np.log(max_support)
    n_support = int(np.exp(np.random.uniform(log_min, log_max)))

    # query_ratio = (n_total - n_support) / n_total
    query_ratio = 0.5  # We keep a fixed query ratio to ensure a meaningful evaluation

    data_np = data.squeeze().cpu().numpy()
    labels_np = labels.squeeze().cpu().numpy()

    X_sup, X_query, y_sup, y_query = train_test_split(
        data_np, labels_np, test_size=query_ratio, stratify=labels_np, random_state=None  # To vary at each iteration
    )

    X_sup = torch.from_numpy(X_sup).to(device)
    X_query = torch.from_numpy(X_query).to(device)
    y_sup = torch.from_numpy(y_sup).to(device)
    y_query = torch.from_numpy(y_query).to(device)

    return X_sup, X_query, y_sup, y_query


def get_sizes_per_class(class_choice: str, y_train: np.ndarray, num_classes: int, context_length: int):
    assert num_classes <= 10, "can only handle up to 10 classes"

    if class_choice == "equal":
        min_num_items = 2
        sizes_per_class = [
            max(int(context_length * sum(y_train == c) / len(y_train)), min_num_items) for c in range(num_classes)
        ]

        # add the remaining in case of leftovers
        residual = context_length - sum(sizes_per_class)
        if residual <= 0:
            max_idx = np.argmax(np.array(sizes_per_class))
            sizes_per_class[max_idx] += residual
        else:
            min_idx = np.argmin(np.array(sizes_per_class))
            sizes_per_class[min_idx] += residual
    elif class_choice == "balance":
        sizes_per_class = [context_length // num_classes] * num_classes
        # add the remaining in case of leftovers
        sizes_per_class[0] += context_length - sum(sizes_per_class)
    else:
        raise NotImplementedError

    return sizes_per_class


def get_knn(
    self,
    y_support: Tensor,
    y_query: Tensor,
    ts_support: Tensor,
    ts_query: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Sequence[Tensor]]:

    num_classes = len(torch.unique(y_support))
    ts_support_np = ts_support.detach().cpu().numpy().copy()
    y_support_np = y_support.detach().cpu().numpy().copy()

    faissKNN = MulticlassFaiss(
        embX=ts_support_np,
        X_orig=ts_support_np,  # FIXME: remove duplicate in the class builder
        y=y_support_np,
        metric="L2",
    )
    sizes_per_class = get_sizes_per_class(
        "equal",
        y_support_np,
        num_classes=num_classes,
        context_length=ts_support.shape[1],
    )
    indices_ts_query_nni, y_nni = faissKNN.get_nearest_neighbors(
        embX_query=ts_query.cpu().numpy(),
        sizes_per_class=sizes_per_class,
    )
    ts_nni = np.concatenate(
        [ts_support_np[y_support_np == i][indices] for i, indices in enumerate(indices_ts_query_nni)],
        axis=0,
    )


class SingleclassFaiss:
    def __init__(self, X, y, metric="L2", gpu_id=0):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        self.y = y

        # self.gpu_resources = StandardGpuResources()
        if metric == "L2":
            self.index = faiss.IndexFlatL2(X.shape[1])
        elif metric == "IP":
            self.index = faiss.IndexFlatIP(X.shape[1])
        else:
            raise NotImplementedError

        self.index.add(X)

    def get_knn_indices(self, queries, k):
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        assert isinstance(k, int)

        knns = self.index.search(queries, k)
        indices_Xs = knns[1]
        ys = self.y[indices_Xs]
        return indices_Xs, ys


class MulticlassFaiss:
    def __init__(self, embX, X_orig, y, metric="L2", gpu_id=0, sizes_per_class=None):
        assert isinstance(embX, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        if embX.dtype != np.float32:
            embX = embX.astype(np.float32)

        self.X = X_orig
        self.y = y

        self.sizes_per_class = sizes_per_class

        self.n_classes = len(np.unique(y))
        if metric == "L2":
            self.indexes = [faiss.IndexFlatL2(embX.shape[1]) for _ in range(self.n_classes)]
        elif metric == "IP":
            self.indexes = [faiss.IndexFlatIP(embX.shape[1]) for _ in range(self.n_classes)]
        else:
            raise NotImplementedError

        for index, Xs in zip(self.indexes, [embX[y == i] for i in range(self.n_classes)]):
            index.add(Xs)

    def get_knn_indices(self, queries, ks):
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()

        if isinstance(ks, int):
            ks = [ks] * self.n_classes
        assert len(ks) == self.n_classes, "ks must have the same length as the number of classes"

        knns = [index.search(queries, k) for index, k in zip(self.indexes, ks)]
        indices_Xs = [x[1] for x in knns]
        ys = np.concatenate([np.ones(k) * i for i, k in enumerate(ks)])
        return indices_Xs, ys

    def get_knn(self, queries, ks=None):
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        queries = queries.astype(np.float32)

        if ks is None:
            assert self.sizes_per_class is not None
            ks = self.sizes_per_class

        if isinstance(ks, int):
            ks = [ks] * self.n_classes
        assert len(ks) == self.n_classes, "ks must have the same length as the number of classes"

        knns = [index.search(queries, k) for index, k in zip(self.indexes, ks)]
        indices_Xs = [x[1] for x in knns]
        Xs = [self.X[self.y == i][indices] for i, indices in enumerate(indices_Xs)]
        Xs = np.concatenate(Xs, axis=1)
        ys = [self.y[self.y == i][indices] for i, indices in enumerate(indices_Xs)]
        ys = np.concatenate(ys, axis=1)
        distances = [x[0] for x in knns]
        distances = np.concatenate(distances, axis=1)

        # because TabPFN is seq len first, batch size second...
        swap01 = lambda x: np.swapaxes(x, 0, 1)
        return swap01(Xs), swap01(ys), swap01(distances), indices_Xs

import os
import rowan
import torch
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader

import LMCE.cfusdlog as cfusdlog


def feature_label_from_data(data: dict, residual_func):
    features = []
    labels = []

    r = []
    start = 0
    for j in range(0, len(data['timestamp'])):
        R = rowan.to_matrix(np.array([data['stateEstimate.qw'][j], data['stateEstimate.qx'][j], data['stateEstimate.qy'][j], data['stateEstimate.qz'][j]]))[:, :2]
        R = R.reshape(1, 6)
        r.append(R[0])

    r = np.array(r)
    k = np.array([data['acc.x'][start:], data['acc.y'][start:], data['acc.z'][start:], data['gyro.x'][start:], data['gyro.y'][start:],data['gyro.z'][start:]]).T
    features = np.append(k, r, axis=1)

    f_a, tau_a = residual_func(data)
    # labels = np.append(f_a, tau_a, axis=1)
    labels = np.array([f[:2] for f in f_a])

    return features, labels


def prepare_data(file_paths: list, residual_func, save_as: str="", shuffle_data: bool=True, overwrite: bool=True):
    if os.path.exists(f"./data/{save_as}.npz") and not overwrite:
        print("Data already exists, loading from files...")
        loaded_arrays = np.load(f"./data/{save_as}.npz")
        X = loaded_arrays['X']
        y = loaded_arrays['y']
        return X, y
    
    X = []
    y = []

    for i, file_path in enumerate(file_paths):
        try:
            data_usd = cfusdlog.decode(file_path)
        except:
            print(f"Failed to load {file_path}!")
            continue
        data = data_usd['fixedFrequency']
        features, labels = feature_label_from_data(data, residual_func)
        X.extend(features)
        y.extend(labels)
        print(f"Preparing data... ({((i+1.)/len(file_paths)*100):.2f}% Done)")

    X = np.array(X)
    y = np.array(y)
    if shuffle_data:
        X, y = shuffle(X, y)

    if save_as:
        if not os.path.exists("./data"):
            os.makedirs("./data")
        np.savez(f'./data/{save_as}.npz', X=X, y=y)

    return X, y

def create_dataloader(X: np.ndarray, y: np.ndarray):

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64)
    return dataloader

import os
import sys
import rowan
import torch
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../on_fly_performace_data'))
import cfusdlog # type: ignore
from residual_calculation import residual # type: ignore


def feature_label_from_data(data: dict):
    features = []
    labels = []

    r = []
    for j in range(1, len(data['timestamp'])):
        R = rowan.to_matrix(np.array([data['stateEstimate.qw'][j], data['stateEstimate.qx'][j], data['stateEstimate.qy'][j], data['stateEstimate.qz'][j]]))[:, :2]
        R = R.reshape(1, 6)
        r.append(R[0])

    r = np.array(r)
    k = np.array([data['acc.x'][1:], data['acc.y'][1:], data['acc.z'][1:], data['gyro.x'][1:], data['gyro.y'][1:],data['gyro.z'][1:]]).T
    features = np.append(k, r, axis=1)

    f_a, tau_a, = residual(data)
    labels = np.append(f_a, tau_a, axis=1)

    return features, labels


def prepare_data(file_paths: list, save_as: str="", shuffle_data: bool=True):
    if os.path.exists(f"./{save_as}.npz"):
        print("Data already exists, loading from files...")
        loaded_arrays = np.load(f"./{save_as}.npz")
        X = loaded_arrays['X']
        y = loaded_arrays['y']
        return X, y
    
    X = []
    y = []

    for i, file_path in enumerate(file_paths):

        data_usd = cfusdlog.decode(file_path)
        data = data_usd['fixedFrequency']
        features, labels = feature_label_from_data(data)
        X.extend(features)
        y.extend(labels)
        print(f"Preparing data... ({((i+1.)/len(file_paths)*100):.2f}% Done)")

    X = np.array(X)
    y = np.array(y)
    if shuffle_data:
        X, y = shuffle(X, y)

    if save_as:
        np.savez(f'./{save_as}.npz', X=X, y=y)

    return X, y

def create_dataloader(X: np.ndarray, y: np.ndarray):
    # print("Mins: ", end="")
    # for j in range(6):
    #     print(f"{min(y[:, j]):.7f}, ", end="")
    # print()
    # print("Maxs: ", end="")
    # for j in range(6):
    #     print(f"{max(y[:, j]):.7f}, ", end="")
    # print()

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64)
    return dataloader

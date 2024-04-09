import torch
import cfusdlog
import numpy as np
from sklearn.utils import shuffle
from residual_calculation import residual
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from config.multirotor_config import MultirotorConfig
from torch.utils.data import TensorDataset, DataLoader


d2r = MultirotorConfig.deg2rad
g = MultirotorConfig.GRAVITATION

def decode_data(path):
    data_usd = cfusdlog.decode(path)
    data = data_usd['fixedFrequency']
    return data

def train_test_data(keys=[]):

    X, y = [], []
    for i in ['00', '01', '02', '03', '04', '05', '06', '10', '11', '20', '23', '24', '25', '27', '28', '29', '30', '32', '33']:
        data = decode_data(f"../flight_data/jana{i}")

        x = []
        if len(keys):
            for key in keys:
                x.append(data[key][1:])
        else:
            for key in data.keys():
                if key != "timestamp":
                    x.append(data[key][1:])
        
        x = np.vstack(x).T
        for i in x:
            X.append(i)

        f_a, tau_a, = residual(data)
        tmp = np.append(f_a, tau_a, axis=1)
        y.append(tmp)

    X = np.vstack(X)
    y = np.vstack(y)
    X, y = shuffle(X, y, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
    X = minmax_scaler.fit_transform(X)
    y = minmax_scaler.fit_transform(y)

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    return train_dataloader, test_dataloader, X_test, y_test

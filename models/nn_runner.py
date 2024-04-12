import numpy as np
from data_fetcher import train_test_data
from NeuralNetworks.model import NeuralNetwork
from torch.utils.data import TensorDataset, DataLoader


X_train, y_train, X_test, y_test, minmax_scaler_output = train_test_data(["rotmat", "acc.x", "acc.y", "acc.z", "gyro.x", "gyro.y", "gyro.z"])
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

model = NeuralNetwork(input_size=len(X_test[0]))
model.train_model(train_dataloader, test_dataloader)

pred_arr = []
for i in range(len(y_test)):
    pred = model.forward(X_test[i])
    pred_arr.append(pred.cpu().detach().numpy())
pred_arr = np.array(pred_arr)
y_test = np.array(y_test)

pred_arr = minmax_scaler_output.inverse_transform(pred_arr)
y_test = minmax_scaler_output.inverse_transform(y_test)

error_f = np.abs(y_test-pred_arr)[:,:3]
print(f"f error rows: {np.mean(error_f, axis = 0 )}")
print(f"overall error f: {np.mean(error_f)}")

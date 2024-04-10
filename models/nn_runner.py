import numpy as np
from data_fetcher import train_test_data
from NeuralNetworks.model import NeuralNetwork
from torch.utils.data import TensorDataset, DataLoader


X_train, y_train, X_test, y_test = train_test_data()
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
err = pred_arr-y_test
print(sum(err)/len(err))

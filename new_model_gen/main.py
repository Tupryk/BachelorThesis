import torch
import numpy as np
from model import MLP
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_prep import prepare_data, create_dataloader

# Get train data paths
indices = ['00', '01', '03', '04','05', '06', '10', '11','20', '23', '24', '25', '27', '28', '29', '30', '32', '33']
file_paths = [f"../flight_data/jana{i}" for i in indices]

# Prepare data
X_train, y_train = prepare_data(file_paths, save_as="train_data")
X_test, y_test = prepare_data(["../flight_data/jana02"], save_as="test_data", shuffle_data=False)
# X_test, y_test0 = prepare_data(["../flight_data/jana00"], save_as="jana00", shuffle_data=False)
# X_test, y_test = prepare_data(["../flight_data/jana03"], save_as="jana03", shuffle_data=False)

# fig, ax = plt.subplots(3)
# ax[0].plot(y_test0[:, 0], label="jana00")
# ax[0].plot(y_test[:, 0], label="jana03")
# ax[0].set_title('Force X')

# ax[1].plot(y_test0[:, 1], label="jana00")
# ax[1].plot(y_test[:, 1], label="jana03")
# ax[1].set_title('Force Y')

# ax[2].plot(y_test0[:, 2], label="jana00")
# ax[2].plot(y_test[:, 2], label="jana03")
# ax[2].set_title('Force Z')

# ax[0].legend()
# plt.tight_layout()
# plt.show()
# exit()

# y_full = np.append(y_train, y_test, axis = 0)
# minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
# y_scaled = minmax_scaler.fit_transform(y_full)
# y_train = y_scaled[:len(y_train),:]
# y_test = y_scaled[len(y_train):,:]

train_dataloader = create_dataloader(X_train, y_train)
test_dataloader = create_dataloader(X_test, y_test)

# Create and train neural network
model = MLP()
model.train_model(train_dataloader, test_dataloader, epochs=32)
model.show_progress()
model.save()
# model.load_state_dict(torch.load('model.pth'))

# Plot results
model.double()
tensor_input = torch.from_numpy(X_test).double()
pred = model.forward(tensor_input).detach().numpy()

# Residual forces
fig, ax = plt.subplots(3)
ax[0].plot(y_test[:, 0], label="Real")
ax[0].plot(pred[:, 0], label="Predicted")
ax[0].set_title('Force X')

ax[1].plot(y_test[:, 1], label="Real")
ax[1].plot(pred[:, 1], label="Predicted")
ax[1].set_title('Force Y')

ax[2].plot(y_test[:, 2], label="Real")
ax[2].plot(pred[:, 2], label="Predicted")
ax[2].set_title('Force Z')

ax[0].legend()
plt.tight_layout()
plt.show()

# Residual torques
fig, ax = plt.subplots(3)
ax[0].plot(y_test[:, 3], label="Real")
ax[0].plot(pred[:, 3], label="Predicted")
ax[0].set_title('Torque X')

ax[1].plot(y_test[:, 4], label="Real")
ax[1].plot(pred[:, 4], label="Predicted")
ax[1].set_title('Torque Y')

ax[2].plot(y_test[:, 5], label="Real")
ax[2].plot(pred[:, 5], label="Predicted")
ax[2].set_title('Torque Z')

ax[0].legend()
plt.tight_layout()
plt.show()

import pickle
import numpy as np
# import xgboost as xg
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from data_prep import prepare_data

# Get train data paths
indices = ['00', '01', '03', '04','05', '06', '10', '11','20', '23', '24', '25', '27', '28', '29', '30', '32', '33']
file_paths = [f"../flight_data/jana{i}" for i in indices]

# Prepare data
X_train, y_train = prepare_data(file_paths, save_as="train_data")
X_test, y_test = prepare_data(["../flight_data/jana02"], save_as="test_data", shuffle_data=False)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=5, verbose=1, max_depth=10, min_samples_split=200, ccp_alpha=0.0))
model.fit(X_train, y_train)

with open('random_forest_regressor_small2.pkl', 'wb') as f:
    pickle.dump(model, f)

del model

with open('random_forest_regressor_small2.pkl', 'rb') as f:
    model = pickle.load(f)

pred = model.predict(X_test)

# eval_set = [(X_train, y_train), (X_test, y_test)]
# model = xg.XGBRegressor(n_estimators=1)
# model.fit(X_train, y_train, eval_set=eval_set)
# pred = model.predict(X_test)
# model.save_model('tree.json')

# Residual forces
fig, ax = plt.subplots(2)
ax[0].plot(y_test[:, 0], label="Real")
ax[0].plot(pred[:, 0], label="Predicted")
ax[0].set_title('Force X')

ax[1].plot(y_test[:, 1], label="Real")
ax[1].plot(pred[:, 1], label="Predicted")
ax[1].set_title('Force Y')

error = np.abs(y_test-pred)
print(f"Average error: (x) -> {np.mean(error[:, 0])} (y) -> {np.mean(error[:, 1])}")

# ax[2].plot(y_test[:, 2], label="Real")
# ax[2].plot(pred[:, 2], label="Predicted")
# ax[2].set_title('Force Z')

ax[0].legend()
plt.tight_layout()
plt.show()

# Residual torques
# fig, ax = plt.subplots(3)
# ax[0].plot(y_test[:, 3], label="Real")
# ax[0].plot(pred[:, 3], label="Predicted")
# ax[0].set_title('Torque X')

# ax[1].plot(y_test[:, 4], label="Real")
# ax[1].plot(pred[:, 4], label="Predicted")
# ax[1].set_title('Torque Y')

# ax[2].plot(y_test[:, 5], label="Real")
# ax[2].plot(pred[:, 5], label="Predicted")
# ax[2].set_title('Torque Z')

# ax[0].legend()
# plt.tight_layout()
# plt.show()


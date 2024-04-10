import numpy as np
import xgboost as xg
import matplotlib.pyplot as plt
from data_fetcher import train_test_data


X_train, y_train, X_test, y_test = train_test_data(keys=["gyro.x"])
y_train = y_train[:, :3]
y_test = y_test[:, :3]

eval_set = [(X_train, y_train), (X_test, y_test)]

model = xg.XGBRegressor(n_estimators=2, objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set)

pred = model.predict([10])
pred = np.array(pred)
print(pred)

model.save_model('forest.json')
xg.plot_tree(model)
plt.savefig("tree.jpg", dpi=1000)

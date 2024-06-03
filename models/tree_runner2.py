from data_fetcher import train_test_data
from sklearn.ensemble import RandomForestRegressor


X_train, y_train, X_test, y_test, _ = train_test_data(keys=["gyro.x"])
y_train = y_train[:, :1]
y_test = y_test[:, :1]

regressor = RandomForestRegressor(random_state=42, n_estimators=1, max_leaf_nodes=6)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(y_pred)

first_tree = regressor.estimators_[0]
leaf_node_values = first_tree.tree_.value.squeeze()
print("Leaf node values:")
for l in leaf_node_values:
    print(l)

print("Shared values: ")
not_shared = 0
shared = 0
for p in y_pred:
    if p in leaf_node_values:
        shared += 1
    else:
        not_shared += 1

print("shared: ", shared)
print("not shared: ", not_shared)

import json
import numpy as np
import xgboost as xg
from data_fetcher import train_test_data
import matplotlib.pyplot as plt

eval_set = [(X_train, y_train), (X_test, y_test)]

model = xg.XGBRegressor(n_estimators=1, objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set)
model.save_model('./forest.json')
xg.plot_tree(model)
plt.savefig("tree.jpg", dpi=1000)

data = json.load(open('./forest.json', "r"))
trees_list = data["learner"]["gradient_booster"]["model"]["trees"]
leaf_node_values = trees_list[0]["split_conditions"]

y_pred = model.predict(X_test)
print("Shared values: ")
not_shared = 0
shared = 0
for p in y_pred:
    for i, l in enumerate(leaf_node_values):
        if np.abs(p - l) < 1e-6:
            shared += 1
            break
        if i == len(leaf_node_values)-1:
            not_shared += 1

print("shared: ", shared)
print("not shared: ", not_shared)

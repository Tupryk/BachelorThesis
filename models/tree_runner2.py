from data_fetcher import train_test_data
from sklearn.ensemble import RandomForestRegressor


X_train, y_train, X_test, y_test = train_test_data(keys=["gyro.x"])
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


import xgboost as xg
from data_fetcher import train_test_data


X_train, y_train, X_test, y_test = train_test_data(keys=["rotmat", "gyro.x"])

eval_set = [(X_train, y_train), (X_test, y_test)]
model = xg.XGBRegressor(n_estimators=100, objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set)

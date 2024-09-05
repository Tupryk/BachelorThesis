# Mostly ChatGPT
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from data_prep import prepare_data

from sklearn.metrics import mean_squared_error

# Get train data paths
indices = ['00', '01', '03', '04','05', '06', '10', '11','20', '23', '24', '25', '27', '28', '29', '30', '32', '33']
file_paths = [f"../flight_data/jana{i}" for i in indices]

# Prepare data
X_train, y_train = prepare_data(file_paths, save_as="train_data")
X_test, y_test = prepare_data(["../flight_data/jana02"], save_as="test_data", shuffle_data=False)

reg = DecisionTreeRegressor(random_state=0, min_samples_split=500, max_depth=10)
path = reg.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas  # Get different alpha values
impurities = path.impurities   # Corresponding impurities for each alpha

# Initialize an empty list to store cross-validation scores
cv_scores = []

# Perform cross-validation for each alpha value
for i, alpha in enumerate(ccp_alphas):
    print((i/len(ccp_alphas))*100, "%")
    # Initialize the DecisionTreeRegressor with the current alpha value
    reg = DecisionTreeRegressor(random_state=0, max_depth=10, min_samples_split=500, ccp_alpha=alpha)
    
    # Perform cross-validation and calculate the mean score (negative MSE)
    scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(scores.mean())

# Find the index of the best alpha value (the one with the lowest cross-validation error)
best_alpha_idx = cv_scores.index(max(cv_scores))
best_alpha = ccp_alphas[best_alpha_idx]

print(f"Best ccp_alpha: {best_alpha}")

# Train the final model with the best ccp_alpha
reg_best = DecisionTreeRegressor(random_state=0, max_depth=10, min_samples_split=500, ccp_alpha=best_alpha)
reg_best.fit(X_train, y_train)

# Evaluate the final model on the test set
y_pred = reg_best.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test Mean Squared Error of pruned tree: {test_mse}")

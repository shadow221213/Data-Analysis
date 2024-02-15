# <div align="center">XGBoost</div>

``` python
# Set up the XGBoost classifier with default hyperparameters
xgb_params = {
        'n_estimators':     500,
        'learning_rate':    0.05,
        'max_depth':        7,
        'subsample':        1.0,
        'colsample_bytree': 1.0,
        'n_jobs':           -1,
        'eval_metric':      'logloss',
        'objective':        'binary:logistic',
        'verbosity':        0,
        'random_state':     1,
}
model = xgb.XGBClassifier(**xgb_params)

# Define the hyperparameters to tune and their search ranges
param_dist = {
        'n_estimators':     np.arange(50, 1000, 50),
        'max_depth':        np.arange(3, 15, 2),
        'learning_rate':    np.arange(0.001, 0.05, 0.004),
        'subsample':        [0.1, 0.3, 0.5, 0.7, 0.9],
        'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
}

# Set up the RandomizedSearchCV object with cross-validation
random_search = RandomizedSearchCV(model, param_distributions = param_dist, cv = 3, n_iter = 50, random_state = 1, n_jobs = -1)
random_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding mean cross-validation score
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

# Evaluate the best model on the test data
best_model = random_search.best_estimator_
print(best_model)
xgb_params = random_search.best_params_
```
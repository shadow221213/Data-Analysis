# <div align="center">LightGBM</div>

``` python
# Set up the LightGBM classifier with default hyperparameters
lgb_params = {
        'n_estimators':     100,
        'max_depth':        7,
        'learning_rate':    0.05,
        'subsample':        0.2,
        'colsample_bytree': 0.56,
        'reg_alpha':        0.25,
        'reg_lambda':       5e-08,
        'objective':        'binary',
        'metric':           'accuracy',
        'boosting_type':    'gbdt',
        'device':           'cpu',
        'random_state':     1,
}
model = lgb.LGBMClassifier(**lgb_params)

# Define the hyperparameters to tune and their search ranges
param_dist = {
        'n_estimators':     np.arange(50, 1000, 50),
        'max_depth':        np.arange(3, 15, 2),
        'learning_rate':    np.arange(0.001, 0.02, 0.002),
        'subsample':        [0.1, 0.3, 0.5, 0.7, 0.9],
        'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
        'reg_alpha':        [uniform(0, 1), uniform(0, 1), uniform(0, 1), uniform(0, 1)],
        'reg_lambda':       [uniform(0, 1), uniform(0, 1), uniform(0, 1), uniform(0, 1)],
}

# Set up the RandomizedSearchCV object with cross-validation
random_search = RandomizedSearchCV(model, param_distributions = param_dist, cv = 3, n_iter = 20,
                                   random_state = 1, n_jobs = -1)

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)

# Print the best hyperparameters and corresponding mean cross-validation score
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

# Evaluate the best model on the test data
best_model = random_search.best_estimator_

lgb_params = random_search.best_params_**
```
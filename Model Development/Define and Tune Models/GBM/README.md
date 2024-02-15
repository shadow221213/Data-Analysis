# <div align="center">GBM</div>

``` python
# Define the hyperparameter search space
param_dist = {
        'n_estimators':      np.arange(100, 1000, 50),
        'learning_rate':     np.logspace(-4, 0, num = 100),
        'max_depth':         [2, 3, 4, 5, 6],
        'min_samples_split': [2, 3, 4, 5, 6],
        'min_samples_leaf':  [1, 2, 3, 4, 5],
        'max_features':      ['sqrt', 'log2', None]
}

# Create the GradientBoostingClassifier model
model = GradientBoostingClassifier(max_depth = 4, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 5, n_estimators = 341, random_state = 42)

# Create the random search object
random_search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_dist,
        n_iter = 100,
        cv = 5,
        scoring = 'accuracy',
        n_jobs = -1,
        random_state = 42
)

# Fit the random search object to the data
random_search.fit(X_train, y_train)

# Print the best parameters and best score
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

gbm_params = random_search.best_params_
```
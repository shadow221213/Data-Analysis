# <div align="center">HistGBM</div>

``` python
# Define the hyperparameter grid to search
param_grid = {
        'learning_rate':     [0.01, 0.05, 0.1, 0.2],
        'max_depth':         [3, 5, 7, 9],
        'max_leaf_nodes':    [15, 31, 63, 127],
        'min_samples_leaf':  [1, 3, 5, 7],
        'l2_regularization': np.logspace(-4, 1, 6),
        'max_bins':          [32, 64, 128, 256],
        'random_state':      [42]
}

# Create a HistGradientBoostingClassifier object
clf = HistGradientBoostingClassifier(max_iter = 2000)

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(
        estimator = clf,
        param_distributions = param_grid,
        n_iter = 30,  # number of parameter settings that are sampled
        cv = 3,  # cross-validation generator
        scoring = 'accuracy',
        n_jobs = -1,
        random_state = 42
)

# Fit the RandomizedSearchCV object on the training data
random_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

hgb_params = random_search.best_params_
```
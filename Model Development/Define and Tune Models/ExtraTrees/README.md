# <div align="center">ExtraTrees</div>

``` python
# Define the hyperparameter space
param_distributions = {
        'n_estimators':      np.arange(100, 1000, 100),
        'max_depth':         [None, 5, 10, 15],
        'max_features':      ['sqrt', 'log2'],
        'min_samples_split': np.arange(2, 10, 2),
        'min_samples_leaf':  np.arange(1, 5, 1),
        'bootstrap':         [True, False]
}

# Define the model
model = ExtraTreesClassifier(random_state = 1)

# Define the random search with cross validation
random_search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_distributions,
        n_iter = 20,
        cv = 5,
        n_jobs = -1,
        random_state = 1
)

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

et_params = random_search.best_params_
```
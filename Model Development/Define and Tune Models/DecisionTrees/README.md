# <div align="center">DecisionTrees</div>

``` python
# Define the hyperparameter space
param_distributions = {
        'max_depth':         np.arange(2, 50, 1),
        'min_samples_split': np.arange(2, 20, 2),
        'min_samples_leaf':  np.arange(1, 10, 1),
        'criterion':         ['gini', 'entropy'],
}

# Define the model
model = DecisionTreeClassifier(random_state = 42)

# Define the random search with cross validation
random_search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_distributions,
        n_iter = 50,
        cv = 5,
        n_jobs = -1,
        random_state = 1
)

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

dtc_params = random_search.best_params_
```
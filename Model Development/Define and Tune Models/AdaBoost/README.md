# <div align="center">AdaBoost</div>

``` python
# Define the hyperparameter space
param_distributions = {
        'n_estimators':  np.arange(50, 500, 50),
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
        'algorithm':     ['SAMME', 'SAMME.R']
}

# Define the model
model = AdaBoostClassifier(random_state = 42)

# Define the random search with cross validation
random_search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_distributions,
        n_iter = 50,
        cv = 5,
        n_jobs = -1,
        random_state = 42
)

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

# Set the best parameters to the model
ada_params = random_search.best_params_
```
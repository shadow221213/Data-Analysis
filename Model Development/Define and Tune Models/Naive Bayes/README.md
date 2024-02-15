# <div align="center">Naive Bayes</div>

``` python
# Define the hyperparameter space
param_distributions = {
        'var_smoothing': np.arange(1e-10, 1e-8, 1e-9)
}

# Define the model
model = GaussianNB( )

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

nb_params = random_search.best_params_
```
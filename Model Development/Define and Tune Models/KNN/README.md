# <div align="center">KNN</div>

``` python
# Define the hyperparameter space
param_distributions = {
        'n_neighbors': np.arange(2, 20, 2),
        'weights':     ['uniform', 'distance'],
        'algorithm':   ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size':   randint(1, 100),
        'p':           [1, 2],
}

# Define the model
model = KNeighborsClassifier( )

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

knn_params = random_search.best_params_
```
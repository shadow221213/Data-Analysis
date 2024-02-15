# <div align="center">Random Forests</div>

``` python
# define the hyperparameter search space
param_distributions = {
        'n_estimators':      [100, 200, 300, 400, 500],
        'max_depth':         [3, 4, 5, 6, 7, 8, 9, 10, None],
        'max_features':      ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
        'bootstrap':         [True, False]
}

# create a RandomForestClassifier model with default parameters
model = RandomForestClassifier(bootstrap = False, max_depth = 4, max_features = 'sqrt', min_samples_leaf = 2, min_samples_split = 5, n_estimators = 341, random_state = 42)

# perform random search with cross-validation
random_search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_distributions,
        n_iter = 15,  # number of parameter settings that are sampled
        scoring = 'accuracy',  # use accuracy as the evaluation metric
        cv = 5,  # 5-fold cross-validation
        verbose = 1,
        n_jobs = -1,
        random_state = 42
)

# fit the random search object to the training data
random_search.fit(X_train, y_train)

# print the best parameters and best score
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

rf_params = random_search.best_params_
```
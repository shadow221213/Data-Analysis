<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-22 20:30:28
 * @LastEditTime: 2024-01-22 23:24:32
-->
# <div align="center">Logistic</div>

``` python
# define the hyperparameter search space
param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C':       [0.001, 0.01, 0.1, 1, 10, 100],
        'solver':  ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# create a LogisticRegression model with default parameters
model = LogisticRegression(max_iter = 1000, random_state = 2023)

# perform grid search with cross-validation
grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        scoring = 'roc_auc',  # use accuracy as the evaluation metric
        cv = 5,  # 5-fold cross-validation
        verbose = 1,
        n_jobs = -1
)

# fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# print the best parameters and best score
print("Best hyperparameters: ", grid_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(grid_search.best_score_))

lg_params = grid_search.best_params_
```
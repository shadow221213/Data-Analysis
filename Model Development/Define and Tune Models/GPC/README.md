<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-22 22:07:35
 * @LastEditTime: 2024-01-22 22:07:52
-->
# <div align="center">GPC</div>

``` python
# Define the hyperparameter space
param_distributions = {
        "kernel":           [1.0 * RBF(l) for l in uniform(0.01, 10).rvs(10)],
        "optimizer":        ["fmin_l_bfgs_b", "fmin_tnc", "fmin_powell"],
        "max_iter_predict": randint(10, 500),
}

# Define the model
model = GaussianProcessClassifier(random_state = 1)

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
random_search.fit(X_train[:1000], y_train[:1000])

# Print the best parameters and score
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

gpc_params = random_search.best_params_
```
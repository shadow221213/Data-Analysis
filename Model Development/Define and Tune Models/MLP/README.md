<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-22 22:06:21
 * @LastEditTime: 2024-01-22 23:24:50
-->
# <div align="center">MLP</div>

``` python
# Define the hyperparameter space
param_distributions = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
        'activation':         ['logistic', 'tanh', 'relu'],
        'solver':             ['lbfgs', 'adam'],
        'alpha':              uniform(0.0001, 0.1),
        'learning_rate':      ['constant', 'invscaling', 'adaptive'],
        'learning_rate_init': uniform(0.0001, 0.1),
}

# Define the model
model = MLPClassifier(random_state = 42, max_iter = 1000)

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

mlp_params = random_search.best_params_
```
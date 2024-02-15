<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-22 21:58:32
 * @LastEditTime: 2024-01-22 23:25:13
-->
# <div align="center">SVM</div>

``` python
# Define the hyperparameter space
param_distributions = {
        'C':      uniform(0.1, 10),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': randint(1, 10),
        'gamma':  ['scale', 'auto'] + list(uniform(0.01, 1).rvs(10)),
}

# Define the model
model = SVC(probability = True)

# Define the random search with cross validation
random_search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_distributions,
        n_iter = 5,
        cv = 3,
        n_jobs = -1,
        random_state = 1
)

# Fit the random search to the data
random_search.fit(X_train[:1000], y_train[:1000])

# Print the best parameters and score
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

svm_params = random_search.best_params_
```
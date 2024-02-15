# <div align="center">CatBoost</div>

``` python
# define the hyperparameter search space
param_distributions = {
        'depth':            np.arange(3, 15, 2),
        'learning_rate':    np.arange(0.001, 0.02, 0.002),
        'l2_leaf_reg':      [0.1, 0.5, 0.7],
        'random_strength':  [0.1, 0.2, 0.3],
        'max_bin':          [100, 150, 200],
        'grow_policy':      ['SymmetricTree', 'Depthwise', 'Lossguide'],
        'bootstrap_type':   ['Bayesian', 'Bernoulli'],
        'one_hot_max_size': [10, 50, 70],
}

# create a CatBoostClassifier model with default parameters
model = CatBoostClassifier(iterations = 200, eval_metric = 'Accuracy', loss_function = 'Logloss', task_type = 'CPU')

# perform random search with cross-validation
random_search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_distributions,
        n_iter = 50,  # number of parameter settings that are sampled
        scoring = 'neg_log_loss',  # use negative log-loss as the evaluation metric
        cv = 3,  # 5-fold cross-validation
        verbose = 1,
        random_state = 42
)

# fit the random search object to the training data
random_search.fit(X_train, y_train, verbose = 0)

# print the best parameters and best score
print("Best hyperparameters: ", random_search.best_params_)
print("Best mean cross-validation score: {:.3f}".format(random_search.best_score_))

cat_params = random_search.best_params_ 
```
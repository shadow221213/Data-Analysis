# <div align="center">选择模型</div>

``` python
class Splitter:
    
    def __init__( self, kfold = True, n_splits = 5 ):
        self.n_splits = n_splits
        self.kfold = kfold
    
    def split_data( self, X, y, random_state_list ):
        if self.kfold:
            for random_state in random_state_list:
                kf = StratifiedKFold(n_splits = self.n_splits, random_state = random_state, shuffle = True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
        else:
            X_train, X_val = X.iloc[:int(X_train.shape[0] / 10)], X.iloc[int(X_train.shape[0] / 10):]
            y_train, y_val = y.iloc[:int(X_train.shape[0] / 10)], y.iloc[int(X_train.shape[0] / 10):]
            yield X_train, X_val, y_train, y_val

class Classifier:
    
    def __init__( self, n_estimators = 100, device = "cpu", random_state = 0 ):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model( )
        self.len_models = len(self.models)
    
    def _define_model( self ):
        xgb_params.update({
                'objective':    'binary:logistic',
                'n_jobs':       -1,
                'random_state': self.random_state,
        })
        if self.device == 'gpu':
            xgb_params.update({
                    'tree_method': 'gpu_hist',
                    'predictor':   'gpu_predictor',
            })
        
        lgb_params.update({
                'objective':    'binary',
                'random_state': self.random_state,
        })
        
        cat_params.update({
                'task_type':    self.device.upper( ),
                'random_state': self.random_state,
        })
        
        cat_sym_params = cat_params.copy( )
        cat_sym_params['grow_policy'] = 'SymmetricTree'
        cat_dep_params = cat_params.copy( )
        cat_dep_params['grow_policy'] = 'Depthwise'
        dt_params = {'min_samples_split': 80, 'min_samples_leaf': 30, 'max_depth': 8, 'criterion': 'gini'}
        
        models = {
                'ann':     ann,
                'xgb':     xgb.XGBClassifier(**xgb_params),
                'lgb':     lgb.LGBMClassifier(**lgb_params),
                'cat':     CatBoostClassifier(**cat_params),
                "cat_sym": CatBoostClassifier(**cat_sym_params),
                "cat_dep": CatBoostClassifier(**cat_dep_params),
                'lg':      LogisticRegression(**lg_params),
                'rf':      RandomForestClassifier(**rf_params, random_state = self.random_state),
                'hgb':     HistGradientBoostingClassifier(**hgb_params),
                'gbm':     GradientBoostingClassifier(**gbm_params, loss = 'deviance', n_iter_no_change = 300,
                                                      random_state = self.random_state),
                'svc':     SVC(**svm_params, probability = True),
                'knn':     KNeighborsClassifier(**knn_params),
                'mlp':     MLPClassifier(**mlp_params, random_state = self.random_state, max_iter = 1000),
                #             'gpc': GaussianProcessClassifier(**gpc_params, random_state=self.random_state),
                'et':      ExtraTreesClassifier(**et_params, random_state = self.random_state),
                'dt':      DecisionTreeClassifier(**dt_params, random_state = self.random_state),
                'ada':     AdaBoostClassifier(**ada_params, random_state = self.random_state),
                'nb':      GaussianNB(**nb_params)
        }
        
        return models
```
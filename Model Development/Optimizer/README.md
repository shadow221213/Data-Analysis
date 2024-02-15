# <div align="center">优化器</div>

``` python
!pip install cmaes
!pip install optuna
import optuna

from functools import partial

class OptunaWeights:
    
    def __init__( self, random_state ):
        self.study = None
        self.weights = None
        self.random_state = random_state
    
    def _objective( self, trial, y_true, y_preds ):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(len(y_preds))]
        
        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis = 1, weights = weights)
        
        # Calculate the Recall score for the weighted prediction
        precisions, recalls, thresholds = precision_recall_curve(y_true, weighted_pred)
        cutoff = acc_cutoff(y_true, weighted_pred)
        
        y_weight_pred = np.where(weighted_pred > float(cutoff), 1, 0)
        score = metrics.accuracy_score(y_true, y_weight_pred)
        return score
    
    def fit( self, y_true, y_preds, n_trials = 2000 ):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed = self.random_state)
        self.study = optuna.create_study(sampler = sampler, study_name = "OptunaWeights",
                                         direction = 'maximize')
        
        objective_partial = partial(self._objective, y_true = y_true, y_preds = y_preds)
        self.study.optimize(objective_partial, n_trials = n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]
    
    def predict( self, y_preds ):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis = 1, weights = self.weights)
        return weighted_pred
    
    def fit_predict( self, y_true, y_preds, n_trials = 2000 ):
        self.fit(y_true, y_preds, n_trials = n_trials)
        return self.predict(y_preds)
    
    def weights( self ):
        return self.weights

def acc_cutoff_class( y_valid, y_pred_valid ):
    y_valid = np.array(y_valid)
    y_pred_valid = np.array(y_pred_valid)
    
    fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred_valid)
    pred_valid = pd.DataFrame({'label': y_pred_valid})
    thresholds = np.array(threshold)
    
    pred_labels = (pred_valid['label'].values > thresholds[:, None]).astype(int)
    acc_scores = (pred_labels == y_valid).mean(axis = 1)
    
    acc_df = pd.DataFrame({'threshold': threshold, 'test_acc': acc_scores})
    acc_df.sort_values(by = 'test_acc', ascending = False, inplace = True)
    cutoff = acc_df.iloc[0, 0]
    
    y_pred_valid = np.where(y_pred_valid < float(cutoff), 0, 1)
    return y_pred_valid
```
# <div align="center">混合特征分析</div>

``` python
from itertools import combinations

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.model_selection import KFold

# Below are the functions to decide the decision boundaries in order to maximize Accuracy/ f1-score
def acc_cutoff( y_valid, y_pred_valid ):
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
    return cutoff

feature_pairs = list(combinations(num_feat, 2))

table = PrettyTable( )
table.field_names = ['Pair Features', 'Accuracy(CV-TRAIN)', "Selected"]

selected_features = []
max_product = float('-inf')
for pair in feature_pairs:
    col1, col2 = pair
    # print(pair)
    product_col_train = train_data[col1] * train_data[col2]
    product_col_test = test_data[col1] * test_data[col2]
    name = f'{col1}_{col2}_product'
    train_data[name] = product_col_train
    test_data[name] = product_col_test
    max_product = max(max_product, product_col_train.max( ))
    
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    MAE = []
    X = train_data[[name]].values
    y = train_data["Transported"].values
    
    ACC = []
    for train_idx, val_idx in kf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        x_val, y_val = X[val_idx], y[val_idx]
        
        model = LogisticRegression( )
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(x_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred)
        cutoff = acc_cutoff(y_val, y_pred)
        # print(cutoff)
        predicted = pd.DataFrame( )
        predicted["Transported"] = y_pred
        y_pred = np.where(predicted["Transported"] > float(cutoff), 1, 0)
        ACC.append(accuracy_score(y_val, y_pred))
    
    if np.mean(ACC) < 0.7:
        unimportant_features.append(name)
        selected = "No"
    else:
        selected_features.append(pair)
        selected = "Yes"
    
    table.add_row([pair, np.mean(ACC), selected])

table.sortby = 'Accuracy(CV-TRAIN)'
table.reversesort = True

print(table)
```
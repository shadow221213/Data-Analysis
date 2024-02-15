<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-21 23:33:33
 * @LastEditTime: 2024-01-21 23:53:50
-->
# <div align="center">聚类并且热编码</div>

``` python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

!pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer

def OHE( train, test, cols, target ):
    """
    One Hot Encoder
    """
    combined = pd.concat([train, test], axis = 0)
    for col in cols:
        one_hot = pd.get_dummies(combined[col])
        counts = combined[col].value_counts( )
        min_count_category = counts.idxmin( )
        one_hot = one_hot.drop(min_count_category, axis = 1)
        combined = pd.concat([combined, one_hot], axis = "columns")
        combined = combined.drop(col, axis = 1)
        combined = combined.loc[:, ~combined.columns.duplicated( )]
    
    # split back to train and test dataframes
    train_ohe = combined[:len(train)]
    test_ohe = combined[len(train):]
    test_ohe.reset_index(inplace = True, drop = True)
    test_ohe.drop(columns = [target], inplace = True)
    
    return train_ohe, test_ohe

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

table = PrettyTable( )
table.field_names = ['Cluster WOE Feature', 'MAE(CV-TRAIN)']

for col in num_feat:
    sub_set = [f for f in unimportant_features if col in f]
    print(sub_set)
    
    temp_train = train_data[sub_set]
    temp_test = test_data[sub_set]
    
    sc = StandardScaler( )
    temp_train = sc.fit_transform(temp_train)
    temp_test = sc.transform(temp_test)
    model = KMeans( )
    
    # Initialize the KElbowVisualizer with the KMeans model and desired range of clusters
    visualizer = KElbowVisualizer(model, k = (3, 25), metric = 'calinski_harabasz', timings = False)
    visualizer.fit(np.array(temp_train))
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Calinski-Harabasz Index')
    plt.show( )
    
    ideal_clusters = visualizer.elbow_value_
    if ideal_clusters is None:
        ideal_clusters = 25
    print(ideal_clusters)
    
    kmeans = KMeans(n_clusters = ideal_clusters)
    kmeans.fit(np.array(temp_train))
    labels_train = kmeans.labels_
    
    train_data[col + '_OHE_cluster'] = labels_train
    test_data[col + '_OHE_cluster'] = kmeans.predict(np.array(temp_test))
    
    # Also, making a copy to do mean encoding followed by a log transformation
    train_data[col + "_unimp_cluster_WOE"] = train_data[col + '_OHE_cluster']
    test_data[col + "_unimp_cluster_WOE"] = test_data[col + '_OHE_cluster']
    cat_labels = cat_labels = np.log1p(
            train_data.groupby([col + "_unimp_cluster_WOE"])['Transported'].mean( ))
    cat_labels2 = cat_labels.to_dict( )
    train_data[col + "_unimp_cluster_WOE"] = train_data[col + "_unimp_cluster_WOE"].map(cat_labels2)
    test_data[col + "_unimp_cluster_WOE"] = test_data[col + "_unimp_cluster_WOE"].map(cat_labels2)
    
    X = train_data[[col + "_unimp_cluster_WOE"]].values
    y = train_data["Transported"].values
    
    ACC = []
    
    kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
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
    
    table.add_row([col + "_unimp_cluster_WOE", np.mean(ACC)])
    
    train_data[col + '_OHE_cluster'] = col + "_OHE_" + train_data[col + '_OHE_cluster'].astype(str)
    test_data[col + '_OHE_cluster'] = col + "_OHE_" + test_data[col + '_OHE_cluster'].astype(str)
    train_data, test_data = OHE(train_data, test_data, [col + '_OHE_cluster'], "Transported")

print(table)
```
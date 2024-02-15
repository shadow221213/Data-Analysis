<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-21 23:11:42
 * @LastEditTime: 2024-01-21 23:20:05
-->
# <div align="center">编码方法</div>

1. **Count/Frequency Encoding**
2. **Count Labeling**
3. **WOE Binning**
4. **Target-Guided Mean Encoding**
5. **Group Clustering**
6. **One-Hot Encoding**

``` python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

!pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer

cat_features = [*set([feature for feature in train_data.columns if train_data[feature].nunique( ) <= 10 or train_data[feature].dtype == 'O']) - set(
        ["PassengerId", "Transported", "VIP", "CryoSleep"])]
train_data[cat_features].nunique( )

cat_features = ['HomePlanet', 'cabin_deck', 'Destination', 'cabin_side']
table = PrettyTable( )
table.field_names = ['Feature', 'Encoded Feature', "Accuracy (CV)- Logistic regression"]

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

for feature in cat_features:
    ## Target Guided Mean --Data Leakage Possible
    cat_labels = train_data.groupby([feature])['Transported'].mean( ).sort_values( ).index
    cat_labels2 = {k: i for i, k in enumerate(cat_labels, 0)}
    train_data[feature + "_target"] = train_data[feature].map(cat_labels2)
    test_data[feature + "_target"] = test_data[feature].map(cat_labels2)
    
    ## Count Encoding
    dic = train_data[feature].value_counts( ).to_dict( )
    train_data[feature + "_count"] = np.log1p(train_data[feature].map(dic))
    test_data[feature + "_count"] = np.log1p(test_data[feature].map(dic))
    
    ## Count Labeling
    dic2 = train_data[feature].value_counts( ).to_dict( )
    list1 = np.arange(len(dic2.values( )), 0, -1)  # Higher rank for high count
    # list1=np.arange(len(dic2.values())) # Higher rank for low count
    dic3 = dict(zip(list(dic2.keys( )), list1))
    train_data[feature + "_count_label"] = train_data[feature].replace(dic3)
    test_data[feature + "_count_label"] = test_data[feature].replace(dic3)
    
    ## WOE Binning
    cat_labels = np.log1p(train_data.groupby([feature])['Transported'].sum( ) / (train_data.groupby([feature])['Transported'].count( ) - train_data.groupby([feature])['Transported'].sum( )))  # .sort_values().index
    cat_labels2 = cat_labels.to_dict( )
    train_data[feature + "_WOE"] = train_data[feature].map(cat_labels2)
    test_data[feature + "_WOE"] = test_data[feature].map(cat_labels2)
    
    temp_cols = [feature + "_target", feature + "_count", feature + "_count_label", feature + "_WOE"]
    
    # It is possible to have NaN values in the test data when new categories are seen
    imputer = KNNImputer(n_neighbors = 5)
    train_data[temp_cols] = imputer.fit_transform(train_data[temp_cols])
    test_data[temp_cols] = imputer.transform(test_data[temp_cols])
    
    if train_data[feature].dtype != "O":
        temp_cols.append(feature)
    else:
        train_data.drop(columns = [feature], inplace = True)
        test_data.drop(columns = [feature], inplace = True)
        
    # Also, doing a group clustering on all encoding types and an additional one-hot on the clusters
    temp_train = train_data[temp_cols]
    temp_test = test_data[temp_cols]
    
    sc = StandardScaler( )
    temp_train = sc.fit_transform(temp_train)
    temp_test = sc.transform(temp_test)
    model = KMeans( )
    
    # Initialize the KElbowVisualizer with the KMeans model and desired range of clusters
    visualizer = KElbowVisualizer(model, k = (3, 15), metric = 'calinski_harabasz', timings = False)
    visualizer.fit(np.array(temp_train))
    
    ideal_clusters = visualizer.elbow_value_
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Calinski-Harabasz Index')
    plt.title("Clustering on encoded featured from " + feature)
    plt.show( )
    print(ideal_clusters)
    
    if ideal_clusters is not None:
        kmeans = KMeans(n_clusters = ideal_clusters)
        kmeans.fit(np.array(temp_train))
        labels_train = kmeans.labels_
        
        train_data[feature + '_cat_cluster_WOE'] = labels_train
        test_data[feature + '_cat_cluster_WOE'] = kmeans.predict(np.array(temp_test))
        
        train_data[feature + '_cat_OHE_cluster'] = feature + "_OHE_" + train_data[feature + '_cat_cluster_WOE'].astype(str)
        test_data[feature + '_cat_OHE_cluster'] = feature + "_OHE_" + test_data[feature + '_cat_cluster_WOE'].astype(str)
        
        train_data, test_data = OHE(train_data, test_data, [feature + '_cat_OHE_cluster'], "Transported")
        
        cat_labels = cat_labels = np.log1p(train_data.groupby([feature + '_cat_cluster_WOE'])['Transported'].mean( ))
        cat_labels2 = cat_labels.to_dict( )
        train_data[feature + '_cat_cluster_WOE'] = train_data[feature + '_cat_cluster_WOE'].map(cat_labels2)
        test_data[feature + '_cat_cluster_WOE'] = test_data[feature + '_cat_cluster_WOE'].map(cat_labels2)
        
        temp_cols = temp_cols + [feature + '_cat_cluster_WOE']
    else:
        print("No good clusters were found, skipped without clustering and OHE")
    
    # See which transformation along with the original is giving you the best univariate fit with target
    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
    
    accuaries = []
    
    for f in temp_cols:
        X = train_data[[f]].values
        y = train_data["Transported"].values
        
        acc = []
        
        for train_idx, val_idx in skf.split(X, y):
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
            acc.append(accuracy_score(y_val, y_pred))
            
        accuaries.append((f, np.mean(acc)))
        
    best_col, best_acc = sorted(accuaries, key = lambda x: x[1], reverse = True)[0]
    
    # check correlation between best_col and other columns and drop if correlation >0.75
    corr = train_data[temp_cols].corr(method = 'pearson')
    corr_with_best_col = corr[best_col]
    cols_to_drop = [f for f in temp_cols if corr_with_best_col[f] > 0.75 and f != best_col]
    final_selection = [f for f in temp_cols if f not in cols_to_drop]
    
    if cols_to_drop:
        train_data = train_data.drop(columns = cols_to_drop)
        test_data = test_data.drop(columns = cols_to_drop)
    table.add_row([feature, best_col, best_acc])
    
print(table)
```
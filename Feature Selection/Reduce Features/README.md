# <div align="center">减少特征</div>

``` python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

!pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer

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

final_drop_list = []

table = PrettyTable( )
table.field_names = ['Original', 'Final Transformed feature', "Accuray(CV)- Logistic Regression"]

threshold = 0.8
# It is possible that multiple parent features share same child features, so storing selected features to 
# avoid selecting the same feature again
best_cols = []

for col in num_feat:
    sub_set = [f for f in num_derived_list if col in f]
    # print(sub_set)
    
    correlated_features = []
    
    # Loop through each feature
    for i, feature in enumerate(sub_set):
        # Check correlation with all remaining features
        for j in range(i + 1, len(sub_set)):
            correlation = np.abs(train_data[feature].corr(train_data[sub_set[j]]))
            
            # If correlation is greater than threshold, add to list of highly correlated features
            if correlation > threshold:
                correlated_features.append(sub_set[j])
    
    # Remove duplicate features from the list
    correlated_features = list(set(correlated_features))
    if len(correlated_features) > 1:
        temp_train = train_data[correlated_features]
        temp_test = test_data[correlated_features]
        
        # Scale before applying PCA
        sc = StandardScaler( )
        temp_train = sc.fit_transform(temp_train)
        temp_test = sc.transform(temp_test)
        
        # Initiate PCA
        pca = TruncatedSVD(n_components = 1)
        x_pca_train = pca.fit_transform(temp_train)
        x_pca_test = pca.transform(temp_test)
        x_pca_train = pd.DataFrame(x_pca_train, columns = [col + "_pca_comb_final"])
        x_pca_test = pd.DataFrame(x_pca_test, columns = [col + "_pca_comb_final"])
        train_data = pd.concat([train_data, x_pca_train], axis = 'columns')
        test_data = pd.concat([test_data, x_pca_test], axis = 'columns')
        
        # Clustering
        model = KMeans( )
        
        # Initialize the KElbowVisualizer with the KMeans model and desired range of clusters
        visualizer = KElbowVisualizer(model, k = (10, 25), metric = 'calinski_harabasz', timings = False)
        visualizer.fit(np.array(temp_train))
        
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Calinski-Harabasz Index')
        plt.title("Clustering on features from " + col)
        plt.show( )
        
        ideal_clusters = visualizer.elbow_value_
        
        if ideal_clusters is None:
            ideal_clusters = 10
        
        # print(ideal_clusters)
        kmeans = KMeans(n_clusters = ideal_clusters)
        kmeans.fit(np.array(temp_train))
        labels_train = kmeans.labels_
        
        train_data[col + '_final_cluster'] = labels_train
        test_data[col + '_final_cluster'] = kmeans.predict(np.array(temp_test))
        
        cat_labels = cat_labels = np.log1p(
                train_data.groupby([col + "_final_cluster"])['Transported'].mean( ))
        cat_labels2 = cat_labels.to_dict( )
        train_data[col + "_final_cluster"] = train_data[col + "_final_cluster"].map(cat_labels2)
        test_data[col + "_final_cluster"] = test_data[col + "_final_cluster"].map(cat_labels2)
        
        correlated_features = correlated_features + [col + "_pca_comb_final", col + "_final_cluster"]
        # See which transformation along with the original is giving you the best univariate fit with target
        kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
        
        ACC = []
        
        for f in correlated_features:
            X = train_data[[f]].values
            y = train_data["Transported"].values
            
            acc = []
            for train_idx, val_idx in kf.split(X, y):
                X_train, y_train = X[train_idx], y[train_idx]
                x_val, y_val = X[val_idx], y[val_idx]
                
                model = LogisticRegression( )
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(x_val)[:, 1]
                precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred)
                cutoff = acc_cutoff(y_val, y_pred)
                predicted = pd.DataFrame( )
                predicted["Transported"] = y_pred
                y_pred = np.where(predicted["Transported"] > float(cutoff), 1, 0)
                acc.append(accuracy_score(y_val, y_pred))
            
            if f not in best_cols:
                ACC.append((f, np.mean(acc)))
        
        best_col, best_acc = sorted(ACC, key = lambda x: x[1], reverse = True)[0]
        best_cols.append(best_col)
        
        cols_to_drop = [f for f in correlated_features if f not in best_cols]
        if cols_to_drop:
            final_drop_list = final_drop_list + cols_to_drop
        table.add_row([col, best_col, best_acc])
    
    else:
        print(f"All features for {col} have correlation less than threshold")
        table.add_row([col, "All features selected", "--"])

print(table)

final_drop_list = [f for f in final_drop_list if f not in cont_cols]
train_data.drop(columns = [*set(final_drop_list)], inplace = True)
test_data.drop(columns = [*set(final_drop_list)], inplace = True)
```
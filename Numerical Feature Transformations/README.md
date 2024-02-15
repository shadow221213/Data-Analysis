<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-21 21:55:58
 * @LastEditTime: 2024-01-21 22:23:14
-->
# <div align="center">数字特征变换</div>

要看哪种变换对每个特征更有效，然后选择它们，这样做的目的是压缩数据。在某些情况下，可能需要拉伸数据。

这些是应用的方法：

1. **Log Transformation**：当数据高度倾斜且方差随均值增大时，这种方法非常有用。

        y = log(x)

2. **Square Root Transformation**：当数据高度倾斜且方差随均值增大时，这种方法非常有用。
        
        y = sqrt(x)

3. **Box-Cox Transformation**：当数据高度倾斜且方差随均值增大时，这种方法非常有用。

        y = [(x^lambda) - 1] / lambda if lambda != 0
        y = log(x) if lambda = 0

4. **Yeo-Johnson Transformation**：当数据高度倾斜且方差随均值增大时，这种方法非常有用。
                            
        y = [(|x|^lambda) - 1] / lambda if x >= 0, lambda != 0
        y = log(|x|) if x >= 0, lambda = 0
        y = -[(|x|^lambda) - 1] / lambda if x < 0, lambda != 2
        y = -log(|x|) if x < 0, lambda = 2

5. **Power Transformation**：当数据高度倾斜且方差随均值增大时，这种方法非常有用。幂值可以是任何值，通常使用统计方法（如 Box-Cox 或 Yeo-Johnson 变换）来确定。
                            
        y = [(x^lambda) - 1] / lambda if method = "box-cox" and lambda != 0
        y = log(x) if method = "box-cox" and lambda = 0
        y = [(x + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x >= 0, lambda != 0
        y = log(x + 1) if method = "yeo-johnson" and x >= 0, lambda = 0
        y = [-(|x| + 1)^lambda - 1] / lambda if method = "yeo-johnson" and x < 0, lambda != 2
        y = -log(|x| + 1) if method = "yeo-johnson" and x < 0, lambda = 2

``` python
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, PowerTransformer

# The rest are discrete/categorical
num_feat = [f for f in train_data.columns if train_data[f].dtype != "O" and train_data[f].nunique( ) > 10]
print(num_feat)

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

sc = MinMaxScaler( )
unimportant_features = []
table = PrettyTable( )
dt_params = {'min_samples_split': 80, 'min_samples_leaf': 30, 'max_depth': 8, 'criterion': 'absolute_error'}
table.field_names = ['Original Feature', 'Original Accuracy(CV-TRAIN)', 'Transformed Feature', 'Tranformed Accuracy(CV-TRAIN)']

for col in num_feat:
    # Log Transformation after MinMax Scaling(keeps data between 0 and 1)
    train_data["log_" + col] = np.log1p(sc.fit_transform(train_data[[col]]))
    test_data["log_" + col] = np.log1p(sc.transform(test_data[[col]]))
    
    # Square Root Transformation
    train_data["sqrt_" + col] = np.sqrt(sc.fit_transform(train_data[[col]]))
    test_data["sqrt_" + col] = np.sqrt(sc.transform(test_data[[col]]))
    
    # Box-Cox transformation
    transformer = PowerTransformer(method = 'box-cox')
    train_data["bx_cx_" + col] = transformer.fit_transform(
            sc.fit_transform(train_data[[col]]) + 1)  # adjusted to make it +ve
    test_data["bx_cx_" + col] = transformer.transform(sc.transform(test_data[[col]]) + 1)
    
    # Yeo-Johnson transformation
    transformer = PowerTransformer(method = 'yeo-johnson')
    train_data["y_J_" + col] = transformer.fit_transform(train_data[[col]])
    test_data["y_J_" + col] = transformer.transform(test_data[[col]])
    
    # Power transformation, 0.25
    power_transform = lambda x: np.power(x, 0.25)
    transformer = FunctionTransformer(power_transform)
    train_data["pow_" + col] = transformer.fit_transform(sc.fit_transform(train_data[[col]]))
    test_data["pow_" + col] = transformer.transform(sc.transform(test_data[[col]]))
    
    # Power transformation, 0.1
    power_transform = lambda x: np.power(x, 0.1)
    transformer = FunctionTransformer(power_transform)
    train_data["pow2_" + col] = transformer.fit_transform(sc.fit_transform(train_data[[col]]))
    test_data["pow2_" + col] = transformer.transform(sc.transform(test_data[[col]]))
    
    # log to power transformation
    train_data["log_pow2" + col] = np.log1p(train_data["pow2_" + col])
    test_data["log_pow2" + col] = np.log1p(test_data["pow2_" + col])
    
    temp_cols = [col, "log_" + col, "sqrt_" + col, "bx_cx_" + col, "y_J_" + col, "pow_" + col, "pow2_" + col, "log_pow2" + col]
    
    # Fill na becaue, it would be Nan if the vaues are negative and a transformation applied on it
    train_data[temp_cols] = train_data[temp_cols].fillna(0)
    test_data[temp_cols] = test_data[temp_cols].fillna(0)
    
    # Apply PCA on  the features and compute an additional column
    pca = TruncatedSVD(n_components = 1)
    x_pca_train = pca.fit_transform(train_data[temp_cols])
    x_pca_test = pca.transform(test_data[temp_cols])
    x_pca_train = pd.DataFrame(x_pca_train, columns = [col + "_pca_comb"])
    x_pca_test = pd.DataFrame(x_pca_test, columns = [col + "_pca_comb"])
    temp_cols.append(col + "_pca_comb")
    # print(temp_cols)
    
    train_data = pd.concat([train_data, x_pca_train], axis = 'columns')
    test_data = pd.concat([test_data, x_pca_test], axis = 'columns')
    
    # See which transformation along with the original is giving you the best univariate fit with target
    kf = KFold(n_splits = 10, shuffle = True, random_state = 42)
    
    ACC = []
    
    for f in temp_cols:
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
            # print(cutoff)
            predicted = pd.DataFrame( )
            predicted["Transported"] = y_pred
            y_pred = np.where(predicted["Transported"] > float(cutoff), 1, 0)
            acc.append(accuracy_score(y_val, y_pred))
            
        ACC.append((f, np.mean(acc)))
        if f == col:
            orig_acc = np.mean(acc)
            
    best_col, best_acc = sorted(ACC, key = lambda x: x[1], reverse = True)[0]
    
    cols_to_drop = [f for f in temp_cols if f != best_col]
    # print(cols_to_drop)
    final_selection = [f for f in temp_cols if f not in cols_to_drop]
    if cols_to_drop:
        unimportant_features = unimportant_features + cols_to_drop
        
    table.add_row([col, orig_acc, best_col, best_acc])
    
print(table)
```
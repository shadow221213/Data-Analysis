<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-02-15 23:50:14
 * @LastEditTime: 2024-02-16 00:05:38
-->
# <div align="center">计算平均准确率</div>

``` python
# Calculate the mean Accuracy score of the ensemble
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

def acc_cutoff( y_valid, y_pred_valid ):
    y_valid = np.array(y_valid)
    y_pred_valid = np.array(y_pred_valid)
    fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred_valid)
    pred_valid = pd.DataFrame({ 'label': y_pred_valid })
    thresholds = np.array(threshold)
    pred_labels = (pred_valid['label'].values > thresholds[:, None]).astype(int)
    acc_scores = (pred_labels == y_valid).mean(axis=1)
    acc_df = pd.DataFrame({ 'threshold': threshold, 'test_acc': acc_scores })
    acc_df.sort_values(by='test_acc', ascending=False, inplace=True)
    cutoff = acc_df.iloc[0, 0]
    return cutoff

mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble Accuracy score {mean_score:.5f} ± {std_score:.5f}')

# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)

for name, mean_weight, std_weight in zip(models.keys( ), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')

precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_pred)
cutoff = acc_cutoff(y_val, y_val_pred)
print(cutoff)
```
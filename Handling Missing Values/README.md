<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-21 20:16:46
 * @LastEditTime: 2024-01-21 21:49:24
-->
# <div align="center">处理缺失值</div>

将字符的缺失值改为Miss_...，将bool转为数字，保留数字的缺失值，并将结果存为整数

根据某种存在一个一定不存在另一个的逻辑，可执行`train_data[col] = np.where(train_data[col].isna(), np.where(train_data["CryoSleep"] == 1, 0, train_data[col]), train_data[col])`来过滤掉部分为nan的值

采用knn插值补全剩余的nan值

``` python
import pandas as pd
from sklearn.impute import KNNImputer

miss_cont = [feature for feature in train_data.columns if train_data[feature].isnull( ).sum( ) > 0 and train_data[feature].dtype != 'O' and feature not in ['Transported']]
print(miss_cont)

imputer = KNNImputer(n_neighbors = 5)
train_data[miss_cont] = imputer.fit_transform(train_data[miss_cont])
test_data[miss_cont] = imputer.transform(test_data[miss_cont])

# Calculate the missing percentages for both train and test data
train_missing_pct = train_data[miss_cont].isnull( ).mean( ) * 100
test_missing_pct = test_data[miss_cont].isnull( ).mean( ) * 100

# Combine the missing percentages for train and test data into a single dataframe
missing_pct_df = pd.concat([train_missing_pct, test_missing_pct], axis = 1, keys = ['Train %', 'Test%'])
print(missing_pct_df)
```
<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-20 23:32:35
 * @LastEditTime: 2024-02-03 16:45:45
-->

# `<div align="center">`两独立样本t检验`</div>`

T 统计量（t_stat）：这是一个衡量两组均值差异的标准化分数。$|T|$ 统计量越大，表示两组均值差异越显著。

P 值（p_val）：P 值表示在零假设成立的情况下，观察到的样本结果或更极端结果的概率。如果 P 值很小（通常小于显著性水平，如0.05），则有足够的证据拒绝零假设，认为两组均值存在显著差异。

p_val < 0.05 的一类说明具有极大的区分度

```python
from prettytable import PrettyTable
from scipy.stats import ttest_ind

def perform_ttest( train, feature_list, target ):
    """
    Performs t-test on a list of independent features for a binary classification problem

    :param train: pandas dataframe containing the training data
    :param feature_list: list of feature names to perform t-test on
    :param target: name of the target variable (binary)
    :return: dictionary containing t-test results
    """
    ttest_results = {}
    table = PrettyTable( )
  
    table.field_names = ['Feature', 't_stat', 'p_val']
  
    for feature in feature_list:
        group_0 = train[train[target] == 0][feature]
        group_1 = train[train[target] == 1][feature]
      
        t_stat, p_val = ttest_ind(group_0, group_1, nan_policy = 'omit')
        table.add_row([feature, t_stat, p_val])
  
    return print(table)

perform_ttest(train_data, cont_cols, 'Transported')
```

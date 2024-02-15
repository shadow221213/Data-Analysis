<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-20 23:32:35
 * @LastEditTime: 2024-01-20 23:57:08
-->
# <div align="center">方差分析</div>

F 统计量（f_stat）：F 统计量是组间均值差异与组内变异之比。它的值越大，表示组间差异相对于组内差异的比例越大。

P 值（p_val）：P 值表示在零假设成立的情况下，观察到的样本结果或更极端结果的概率。如果 P 值小于显著性水平（通常为0.05），则有足够的证据拒绝零假设，认为至少有一组均值存在显著差异。

p_val < 0.05 的一类说明具有极大的区分度

``` python
from prettytable import PrettyTable
from scipy.stats import f_oneway

def perform_anova( train, feature_list, target ):
    """
    Performs ANOVA on a list of independent features for a binary classification problem

    :param train: pandas dataframe containing the training data
    :param feature_list: list of feature names to perform ANOVA on
    :param target: name of the target variable (binary)
    :return: dictionary containing ANOVA results
    """
    anova_results = {}
    table = PrettyTable( )
    
    table.field_names = ['Feature', 'F-statistic', 'p-value']
    
    for feature in feature_list:
        groups = []
        for group_value in train[target].unique( ):
            group = train[train[target] == group_value][feature].dropna( )
            groups.append(group)
        
        f_stat, p_val = f_oneway(*groups)
        table.add_row([feature, f_stat, p_val])
    
    return print(table)

perform_anova(train_data, cont_cols, 'Transported')
```
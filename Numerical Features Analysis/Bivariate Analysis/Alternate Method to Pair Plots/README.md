<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-20 23:32:35
 * @LastEditTime: 2024-01-21 00:05:08
-->
# <div align="center">图分析的代替方法</div>

如果我们有很多特征，就很难直观地绘制和理解所有特征。因此，有一种方法可以告诉我们，在分类任务中，哪一对特征在一起才是真正重要的。

需要注意的是，我们是通过对数据和特征的理解来决定怎样的组合才是更好的特征。由于表中所有最重要的功能都是支出功能，因此不难理解，创建一个总支出组合将是一个更好的选择。

``` python
from itertools import combinations

from prettytable import PrettyTable
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

feature_pairs = list(combinations(cont_cols, 2))
table = PrettyTable( )
table.field_names = ['Feature Pair', 'Accuracy']

# Fill missing values with the mean of the column
imputer = SimpleImputer(strategy = 'mean')
train_imputed = imputer.fit_transform(train_data[cont_cols])

for pair in feature_pairs:
    # Using the entire train data to fit, not a CV because it is time consuming
    x_temp = train_imputed[:, [cont_cols.index(pair[0]), cont_cols.index(pair[1])]]
    y_temp = train_data['Transported']
    model = SVC(gamma = 'auto')
    model.fit(x_temp, y_temp)
    y_pred = model.predict(x_temp)
    acc = accuracy_score(y_temp, y_pred)
    table.add_row([pair, acc])
    
table.sortby = 'Accuracy'
table.reversesort = True

print(table)
```
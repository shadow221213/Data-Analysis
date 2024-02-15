<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-20 23:27:24
 * @LastEditTime: 2024-01-20 23:31:28
-->
# <div align="center">缺失值检测</div>

``` python
import matplotlib.pyplot as plt
import missingno as msno
from prettytable import PrettyTable

table = PrettyTable( )
table.field_names = ['Column Name', 'Data Type', 'Non-Null Count']

for column in train_data.columns:
    data_type = str(train_data[column].dtype)
    non_null_count = train_data[column].count( )
    table.add_row([column, data_type, non_null_count])
print(table)

msno.matrix(train_data)
plt.show( )

msno.matrix(test_data)
plt.show( )
```
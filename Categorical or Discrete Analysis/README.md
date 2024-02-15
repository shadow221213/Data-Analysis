<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-21 20:00:32
 * @LastEditTime: 2024-01-21 20:04:21
-->
# <div align="center">分类/离散分析</div>

一般在0.4-0.6间的数据相关性不强

``` python
import matplotlib.pyplot as plt
import pandas as pd

cat_features = [f for f in train_data.columns if f not in cont_cols + ["PassengerId", "Name", "Transported"] and train_data[f].nunique( ) < 50]
print(cat_features)

target = 'Transported'

# Create subplots for each categorical feature
fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (16, 8))

# Loop through each categorical feature and plot the contingency table in a subplot
for i, col in enumerate(cat_features):
    contingency_table = pd.crosstab(train_data[col], train_data[target], normalize = 'index')
    contingency_table.plot(kind = 'bar', stacked = True, ax = axs[i // 2, i % 2])
    axs[i // 2, i % 2].set_title(f"{col.title( )} Distribution by Target")
    axs[i // 2, i % 2].set_xlabel(col.title( ))
    axs[i // 2, i % 2].set_ylabel("Proportion")

# Adjust spacing between subplots
fig.tight_layout( )

# Show the plot
plt.show( )
```
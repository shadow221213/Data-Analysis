<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-20 23:36:32
 * @LastEditTime: 2024-01-20 23:39:11
-->
# <div align="center">数据在目标中的分布分析</div>

``` python
import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots for each continuous feature
fig, axs = plt.subplots(nrows = len(cont_cols), figsize = (8, 4 * len(cont_cols)))
for i, col in enumerate(cont_cols):
    sns.boxplot(x = 'Transported', y = col, data = train_data, ax = axs[i], palette = 'pastel')
    axs[i].set_title(f'{col.title( )} vs Target', fontsize = 16)
    axs[i].set_xlabel('Transported', fontsize = 14)
    axs[i].set_ylabel(col.title( ), fontsize = 14)
    axs[i].tick_params(axis = 'both', labelsize = 14)
    sns.despine( )

# Adjust spacing between subplots
fig.tight_layout( )

# Display the plot
plt.show( )
```
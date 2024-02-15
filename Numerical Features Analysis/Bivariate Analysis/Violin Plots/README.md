<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-20 23:32:35
 * @LastEditTime: 2024-01-20 23:48:40
-->
# <div align="center">Voilin分布</div>

将pair中的分析得到的数据放入features中进行分析比较

``` python
import matplotlib.pyplot as plt
import seaborn as sns

# Define the numerical features to plot
features = []

# Create a figure with multiple subplots
fig, axs = plt.subplots(1, len(features), figsize = (16, 5))

# Loop through each feature and plot a violin plot on a separate subplot
for i, col in enumerate(features):
    sns.violinplot(x = 'Transported', y = col, data = train_data, ax = axs[i])
    axs[i].set_title(f'{col.title( )} Distribution by Target', fontsize = 14)
    axs[i].set_xlabel('Transported', fontsize = 12)
    axs[i].set_ylabel(col.title( ), fontsize = 12)
    sns.despine( )

# Adjust spacing between subplots
fig.tight_layout( )

# Display the plot
plt.show( )
```
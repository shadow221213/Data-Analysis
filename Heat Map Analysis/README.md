<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-21 20:05:02
 * @LastEditTime: 2024-01-21 20:06:29
-->
# <div align="center">热力图分析</div>

越红表明相关性越高

``` python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

features = [f for f in train_data.columns if train_data[f].astype != 'O' and f not in ['Transported']]
print(features)

corr = train_data[features].corr(numeric_only = True)
plt.figure(figsize = (10, 10), dpi = 300)

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask = mask, cmap = sns.diverging_palette(500, 10, as_cmap = True), annot = True,
            annot_kws = {'size': 7})
plt.title('Train Feature Correlation Matrix\n', fontsize = 25, weight = 'bold')
plt.show( )
```
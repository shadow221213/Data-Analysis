# <div align="center">热力图分析</div>

``` python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

num_derived_list = []
for f1 in train_data.columns:
    for f2 in num_feat:
        if f2 in f1:
            num_derived_list.append(f1)
num_derived_list = [*set(num_derived_list)]

corr = train_data[num_derived_list].corr( )
plt.figure(figsize = (40, 40), dpi = 300)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr, mask = mask, cmap = sns.diverging_palette(500, 10, as_cmap = True), annot = True,
            annot_kws = {'size': 8})
plt.title('Post-Feature Engineering Correlation Matrix\n', fontsize = 10, weight = 'bold')
plt.show( )
```
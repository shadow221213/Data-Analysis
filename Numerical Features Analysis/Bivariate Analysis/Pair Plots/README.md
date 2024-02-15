<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-20 23:32:35
 * @LastEditTime: 2024-01-21 00:09:55
-->
# <div align="center">pair分布</div>

仅看对角线即可，且需要将y轴转为x轴看待，将其中对目标最有区分度的几组数据找出放入Violin分析

``` python
import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(data = train_data, vars = cont_cols, hue = 'Transported')
plt.show()
```
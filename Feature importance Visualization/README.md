<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-02-16 00:15:41
 * @LastEditTime: 2024-02-16 00:17:52
-->
# <div align="center">重要特征可视化</div>

``` python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def visualize_importance( models, feature_cols, title, top=20 ):
    try:
        importances = []
        feature_importance = pd.DataFrame( )
        for i, model in enumerate(models):
            _df = pd.DataFrame( )
            _df["importance"] = model.feature_importances_
            _df["feature"] = pd.Series(feature_cols)
            _df["fold"] = i
            _df = _df.sort_values('importance', ascending=False)
            _df = _df.head(top)
            feature_importance = pd.concat([feature_importance, _df], axis=0, ignore_index=True)
        
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        # display(feature_importance.groupby(["feature"]).mean().reset_index().drop('fold', axis=1))
        plt.figure(figsize=(12, 10))
        sns.barplot(x='importance', y='feature', data=feature_importance, color='skyblue', errorbar='sd')
        plt.xlabel('Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.title(f'{title} Feature Importance [Top {top}]', fontsize=18)
        plt.grid(True, axis='x')
        plt.show( )
    except AttributeError:
        return

for name, models in trained_models.items( ):
    visualize_importance(models, list(X_train.columns), name)
```
# <div align="center">结果</div>

``` python
import numpy as np
import pandas as pd

list_test_preds = np.where(list_test_preds > cutoff, 1, 0).astype(bool)

model_name = ['ann', 'xgb', 'lgb', 'cat', 'cat_sym', 'cat_dep', 'lg', 'rf', 'hgb', 'gbm', 'svc', 'knn', 'mlp', 'et', 'dt', 'ada', 'nb']

merged_predictions = pd.DataFrame( )

for i, preds in enumerate(list_test_preds):
    col_name = model_name[i]
    merged_predictions[col_name] = preds

test_predss = merged_predictions.mode(axis=1, dropna=True)[0].astype(bool)

sub = pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
sub['Transported'] = test_predss

sub.to_csv('./final_predictions.csv', index=False)
sub.head( )

sub["Transported"].value_counts( ) / sub["Transported"].shape[0]

# sub1=pd.read_csv("/kaggle/working/cat_rf_dt_et_ada.csv")
# sub2=pd.read_csv("/kaggle/working/hgb_gbm_svc_knn_mlp_nb.csv")
# sub3=pd.read_csv("/kaggle/working/xgb_lgb_lg.csv")
# sub4=pd.read_csv("/kaggle/working/anns.csv")

# sub_combined=sub1.copy()
# sub_combined['Transported']=sub1['Transported'] | sub2['Transported'] | sub3['Transported'] | sub4["Transported"]

# sub_combined.to_csv('submission_model.csv',index=False)
```
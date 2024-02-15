# <div align="center">去除不重要的特征</div>

``` python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

print("Number of Unimportant Features are ", len(unimportant_features))

test_data.reset_index(inplace = True, drop = True)

for col in cont_cols:
    sub_set = [f for f in unimportant_features if col in f]
    
    existing = [f for f in train_data.columns if f in sub_set]
    temp_train = train_data[existing]
    temp_test = test_data[existing]
    
    sc = StandardScaler( )
    temp_train = sc.fit_transform(temp_train)
    temp_test = sc.transform(temp_test)
    
    pca = TruncatedSVD(n_components = 1)
    x_pca_train = pca.fit_transform(temp_train)
    x_pca_test = pca.transform(temp_test)
    x_pca_train = pd.DataFrame(x_pca_train, columns = [col + "_pca_comb_unimp"])
    x_pca_test = pd.DataFrame(x_pca_test, columns = [col + "_pca_comb_unimp"])
    
    train_data = pd.concat([train_data, x_pca_train], axis = 'columns')
    test_data = pd.concat([test_data, x_pca_test], axis = 'columns')
    
    for f in sub_set:
        if f in train_data.columns and f not in cont_cols:
            train_data = train_data.drop(columns = [f])
            test_data = test_data.drop(columns = [f])

print("Number of Unimportant Features are ", len(unimportant_features))
```
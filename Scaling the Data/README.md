# <div align="center">缩放数据</div>

``` python
from sklearn.preprocessing import StandardScaler

feature_scale = [feature for feature in train_data.columns if feature not in ['PassengerId', 'Transported']]
print(feature_scale)

scaler = StandardScaler( )
train_data[feature_scale] = scaler.fit_transform(train_data[feature_scale])
test_data[feature_scale] = scaler.transform(test_data[feature_scale])

ID = test_data[['PassengerId']]
train_data.drop(columns = ['PassengerId'], inplace = True)
test_data.drop(columns = ['PassengerId'], inplace = True)

X_train = train_data.drop(['Transported'], axis = 1)
y_train = train_data['Transported']

X_test = test_data.copy( )
print(X_train.shape, X_test.shape)
```
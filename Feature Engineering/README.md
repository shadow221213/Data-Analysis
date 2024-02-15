# <div align="center">特征工程</div>

将一些区分度较强的特征合并为一类单独存放

``` python
train_data["expenditure"] = train_data["VRDeck"] + train_data["Spa"] + train_data["RoomService"]
test_data["expenditure"] = test_data["VRDeck"] + test_data["Spa"] + test_data["RoomService"]
```
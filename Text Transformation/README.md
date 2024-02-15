<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-21 22:35:26
 * @LastEditTime: 2024-01-21 22:36:35
-->
# <div align="center">文本变换</div>

``` python
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf( train, test, column, n, p ):
    """
    Text Transformation

    :param train: train data
    :param test: test data
    :param column: column name to be transformed
    :param n: max features
    :param p: target features
    """
    
    vectorizer = TfidfVectorizer(max_features = n)
    vectors_train = vectorizer.fit_transform(train[column])
    vectors_test = vectorizer.transform(test[column])
    
    svd = TruncatedSVD(p)
    x_pca_train = svd.fit_transform(vectors_train)
    x_pca_test = svd.transform(vectors_test)
    tfidf_df_train = pd.DataFrame(x_pca_train)
    tfidf_df_test = pd.DataFrame(x_pca_test)
    
    cols = [(column + "_tfidf_" + str(f)) for f in tfidf_df_train.columns]
    tfidf_df_train.columns = cols
    tfidf_df_test.columns = cols
    train = pd.concat([train, tfidf_df_train], axis = "columns")
    test = pd.concat([test, tfidf_df_test], axis = "columns")
    
    return (train, test)

(train_data, test_data) = tf_idf(train_data, test_data, "Last_Name", 1000, 5)
train_data.drop(columns = ["Name", "Last_Name"], inplace = True)
test_data.drop(columns = ["Name", "Last_Name"], inplace = True)
```
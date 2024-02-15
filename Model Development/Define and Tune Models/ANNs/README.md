<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-22 15:32:31
 * @LastEditTime: 2024-01-23 02:52:50
-->
# <div align="center">ANNs</div>

``` python
class ANNs(nn.Module):
    
    def __init__( self ):
        super(ANNs, self).__init__( )
        self.layer1 = nn.Linear(X_train_tensor.shape[1], 64)
        self.layer2 = nn.Linear(64, 16)
        self.layer3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU(0.1)
        self.sigmoid = nn.Sigmoid( )
    
    def forward( self, x ):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.layer3(x))
        return x

ann = ANNs( )

criterion = nn.BCELoss( )
optimizer = optim.Adam(ann.parameters( ), lr = 0.001, betas = (0.9, 0.999), eps = 1e-07)
```
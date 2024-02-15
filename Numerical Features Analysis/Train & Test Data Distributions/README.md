<!--
 * @Description: 
 * @Author: shadow221213
 * @Date: 2024-01-20 23:32:35
 * @LastEditTime: 2024-01-20 23:37:06
-->
# <div align="center">数据分布</div>

``` python
import matplotlib.pyplot as plt
import seaborn as sns

cont_cols = [f for f in train_data.columns if
             train_data[f].dtype in [float, int] and train_data[f].nunique( ) > 3]
print(cont_cols)

# Create subplots for each continuous column
fig, axs = plt.subplots(len(cont_cols), 2, figsize = (10, 20))

# Loop through each continuous column and plot the histograms
for i, col in enumerate(cont_cols):
    # Determine the range of values to plot
    max_val = max(train_data[col].max( ), test_data[col].max( ))
    min_val = min(train_data[col].min( ), test_data[col].min( ))
    range_val = max_val - min_val
    
    # Determine the bin size and number of bins
    max_size = 40
    bin_size = range_val / max_size
    num_bins_train = round(range_val / bin_size)
    num_bins_test = round(range_val / bin_size)
    
    # Plot the histograms
    sns.histplot(train_data[col], ax = axs[i][0], color = 'blue', kde = True, label = 'Train',
                 bins = num_bins_train)
    sns.histplot(test_data[col], ax = axs[i][1], color = 'red', kde = True, label = 'Test',
                 bins = num_bins_test)
    axs[i][0].set_title(f'Train - {col}')
    axs[i][0].set_xlabel('Value')
    axs[i][0].set_ylabel('Frequency')
    axs[i][1].set_title(f'Test - {col}')
    axs[i][1].set_xlabel('Value')
    axs[i][1].set_ylabel('Frequency')
    axs[i][0].legend( )
    axs[i][1].legend( )

plt.tight_layout( )
plt.show( )
```
# <div align="center">目标分析</div>

``` python
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the proportion of each class
class_counts = train_data['Transported'].value_counts( )
class_proportions = class_counts / train_data.shape[0]
class_proportions = class_proportions.values.tolist( )
class_proportions_str = [f'{prop:.2%}' for prop in class_proportions]

# Set the color palette
colors = sns.color_palette('pastel')[0:len(class_counts)]

# Plot the distribution of the target variable
plt.figure(figsize = (8, 4))
sns.countplot(x = 'Transported', data = train_data, palette = colors)
plt.title('Distribution of Target Variable', fontsize = 16)
plt.xlabel('Transported', fontsize = 14)
plt.ylabel('Count', fontsize = 14)
plt.ylim([0, len(train_data)])

for i, count in enumerate(class_counts):
    plt.text(i, count + 50, class_proportions_str[i], ha = 'center', fontsize = 14, color = 'black')

plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
sns.despine( )
plt.show( )
```
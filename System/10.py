import pandas as pd

df = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
df
print('Iris Dataset is successfully loaded ....')

import seaborn as sns
import matplotlib.pyplot as plt

choice = 1
while(choice != 10):
	print('------------------- Menu ----------------------------')
	print('1.Display information of Iris Dataset')
	print('2.Display Histogram of Sepal Sepal_Length w.r.t. Species')
	
	choice = int(input('Enter your choice: '))
	if choice == 1:

		print('Information of Dataset:\n', df.info)
		print('Shape of Dataset (row x column): ', df.shape)
		print('Columns Name: ', df.columns)
		print('Total elements in dataset:', df.size)
		print('Datatype of attributes (columns):', df.dtypes)
		print('First 5 rows:\n', df.head().T)
		print('Last 5 rows:\n',df.tail().T)
		print('Any 5 rows:\n',df.sample(5).T)

	if choice == 2:
		sns.histplot(x = df['variety'], kde=True)
		# sns.histplot(x = df['sepal.width'], kde=True)
		# sns.histplot(x = df['petal.length'], kde=True)
		# sns.histplot(x = df['petal.width'], kde=True)
		plt.show()
		
		sns.boxplot(x = df['sepal.length'])
		plt.show()
		# sns.boxplot(x = df['sepal.width'])
		# sns.boxplot(x = df['petal.length'])
		# sns.boxplot(x = df['petal.width'])
        
        
	
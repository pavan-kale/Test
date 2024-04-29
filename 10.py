import pandas as pd

df = pd.read_csv('iris.csv')
print('Iris Dataset is successfully loaded ....')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from plotnine import ggplot , aes, geom_histogram
# ggplot(df)+ aes(x="Sepal_Length")+ geom_histogram(bins=15)
# plt.show()
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
		fig, axes = plt.subplots(2,2)
		fig.suptitle('Histogram of 2-variables')
		plt1=ggplot(df) + aes(x="Sepal_LengthCm") + geom_histogram(bins=15)
		sns.barplot(data = plt1,ax=axes[0,0])
		plt.show()
	
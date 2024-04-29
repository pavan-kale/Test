def DetectOutlier(df, var):
	Q1 = df[var].quantile(0.25)
	Q3 = df[var].quantile(0.75)
	IQR = Q3 - Q1
	high, low = Q3+1.5*IQR, Q1-1.5*IQR
	print("Highest allowed in variable:", var, high) 
	print("lowest allowed in variable:", var, low)
	count = df[(df[var] > high) | (df[var] < low)][var].count()
	print('Total outliers in:', var,':',count)
	df = df[((df[var] >= low) & (df[var] <= high))]
	print('Outliers removed in', var)
	return df
def DrawBoxplot(df, msg):
	fig, axes = plt.subplots(1,2)
	fig.suptitle(msg)
	sns.boxplot(data = df, x ='RM', ax=axes[0])
	sns.boxplot(data = df, x ='LSTAT', ax=axes[1])
	fig.tight_layout()
	plt.show()
#read dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('HousingData.csv')
print('Boston dataset is succesfully loaded ..............................')

print(df.head().T)
print(df.columns) 

print(df.isnull().sum())

Column_Name = ['LSTAT','RM']
Output = ['MEDV']
DrawBoxplot(df, 'Before removing Outliers')
print('Identifying overall outliers in Column Name variables.....')
for var in Column_Name:
	df = DetectOutlier(df,var)
df = DetectOutlier(df, 'RM')
DrawBoxplot(df, 'After removing Outliers')

df['MEDV']=df['MEDV'].astype('category')
print(df.dtypes)
df['MEDV']=df['MEDV'].cat.codes
print(df)
print(df.isnull().sum())

import seaborn as sns
import matplotlib.pyplot as plt 
sns.heatmap(df.corr(),annot=True)
plt.show()

x = df[['RM','LSTAT']]
y = df['MEDV']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x, y, test_size= 0.20, random_state = 0)
print('x_train = ',x_train)
print('x_test = ', x_test)
from sklearn.linear_model import LinearRegression 
model=LinearRegression().fit(x_train,y_train)
y_pred=model.predict(x_test)
print('y_pred :' ,y_pred)
print('y_test :' ,y_test)
from sklearn.metrics import mean_absolute_error
print('MAE: ',mean_absolute_error(y_test,y_pred))
print("Model Score: ", model.score(x_test,y_test))

import numpy as np 
features = np.array([[6,19]])
prediction = model.predict(features)
print('Prediction: {}'.format(prediction))

df=pd.read_csv('HousingData.csv')
print("Successfully Reloaded the Boston Housing Dataset*********")
print("Boston Housing Dataset \n",df)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
df = pd.read_csv('Boston.csv')
print(df)

print(df.describe())

print(df.dtypes)

print(df.shape)

def DetectOutlier(df,var):
	Q1 = df[var].quantile(0.25)
	Q3=df[var].quantile(0.75)
	iqr=Q3-Q1
	high, low = Q3+1.5*iq, Q1-1.5*iqr

	print("Highest allowed in variable:", var, high)
	print("Lowest allowed in variable:", var, low)
	count = df[(df[var]>high)|(df[var]<low)][var].count()
	print('Total outlier in:', var,':', count)

	df= df[((df[var]>=low)&(df[var]<= high))]
	print('outlier removed in ', var)
	return df

def DrawBoxplot(df, msg):
	fig, axes - plt.subplots(1,2)
	fig.suptitle(msg)
	sns.boxplot(data=df, x='rm', ax=axes[0])
	sns.boxplot(data = df, x = 'lstat', ax= axes[1])
	fig.tight_layout()
	plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.corr(),annot=True)
plt.show()

x = df[['rm', 'lstat']]
y = df['medv']
from sklearn.model_selection import train_test_split
X_train, X_test, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)
print('X_train=', X_train)
print('X_test=', X_test)

from sklearn.linear_model import LinearRegression
model= LinearRegression().fit(X_train, y_train)
y_pred=model.predict(X_test)
print('y_pred:', y_pred)
































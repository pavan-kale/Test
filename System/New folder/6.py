import pandas as pd
df = pd.read_csv("iris.csv")
print(df)
print(df.dtypes)
print(df.isnull().sum())
df['species'] = df['species'].astype('category')
print(df.dtypes)
df['species'] = df['species'].cat.codes
print(df.dtypes)
print(df.isnull().sum())

#check outliers 

def DetectOuutlier(df,var):
	Q1 = df[var].quantile(0.25)
	Q3 = df[var].quantile(0.75)
	IQR = Q3 - Q1
	high,low = Q3+1.5*IQR, Q1-1.5*IQR
	print("Highest allowed in variable: ",var,high)
	print("Lowest allowed in variable: ",var,low)

	count = df[(df[var] > high) | (df[var] < low)] [var].count()

	print("Total outliers in: ",var,':',count)

import seaborn as sns
sns.boxplot(df['sepal_width'])

def OutlierRemoval(df,var):
	Q1 = df[var].quantile(0.25)
	Q3 = df[var].quantile(0.75)
	IQR = Q3 - Q1
	high,low = Q3+1.5*IQR, Q1-1.5*IQR
	print("Highest allowed in variable: ",var,high)
	print("Lowest allowed in variable: ",var,low)

	count = df[(df[var] > high) | (df[var] < low)] [var].count()
	print('Total outliers in: ',var,':',count)

	df = df[((df[var] >= low) & (df[var] <= high))]
	return df

	print(df.shape)
	df = OutlierRemoval(df,'sepal_width')
	print(df.shape)

import seaborn as sns
sns.heatmap(df.corr(),annot = True)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

X = df.iloc[:, [0,1,2,3]].values
y = df.iloc[:,4].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import BernoulliNB

classifer = BernoulliNB()

classifer.fit(X_train, y_train)

y_pred = classifer.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred, y_test))
print(accuracy_score(y_pred, y_test))

from sklearn.naive_bayes import GaussianNB

classifer1 = GaussianNB()

classifer1.fit(X_train, y_train)

y_pred1 = classifer1.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred1))

import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True)
plt.savefig('confusion.png')

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))





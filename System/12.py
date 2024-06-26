import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('weather.csv')
print('Weather dataset is successfully loaded .....')
print(df)
print(df.columns)
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
df=df.dropna()
print(df.isnull().sum())
df['RainTomorrow'].unique()
Y=df.RainTomorrow
print(Y.head())
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
df['RainTomorrow']=label_encoder.fit_transform(df['RainTomorrow'])
print(df['RainTomorrow'].unique())
label_encoder=preprocessing.LabelEncoder()
df['WindGustDir']=label_encoder.fit_transform(df['WindGustDir'])
print(df['WindGustDir'].unique())
label_encoder=preprocessing.LabelEncoder()
df['RainToday']=label_encoder.fit_transform(df['RainToday'])
print(df['RainToday'].unique())
import seaborn as sns
import matplotlib.pyplot as plt

# sns.heatmap(df.corr(),annot=True,annot_kws={'size':8})
# sns.heatmap(df.corr(),annot=True)
# plt.figure(figsize=(16,9))
# # sns.set(rc={'figure.figsize':(12.12)})
# plt.show()
X=df.drop(['RainTomorrow','WindDir9am','WindDir3pm','WindSpeed9am'],axis='columns')
print(X.head())
from sklearn.model_selection import train_test_spilt
X_train,X_test,Y_train,Y_test=train_test_spilt(X,Y,test_size=0.2,random_state=10)
from sklearn import svm
clf=svm.SVC(kernel='linear')
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test,y_pred))
model=svm.SVC(kernel='poly')
clf.fit(X_train,Y_train)
y_pred=clf.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test,y_pred))
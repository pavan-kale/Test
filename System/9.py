import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
df = pd.read_csv('titanic.csv') 
print(df.describe()) 
print(df.head()) 
print(df.tail()) 
print(df.isnull().sum()) 
#df['Age'].fillna(df.['Age'].median(),inplace=True) 
plt.figure(figsize=(10,7)) 
sns.boxplot(data=df,x='Age') 
plt.show() 
sns.boxplot(data=df,x='Age',y='Sex') 
plt.show() 
sns.boxplot(data=df,x='Age',y='Sex', hue='Survived') 
plt.show()
import pandas as pd
import numpy as np
df = pd.read_csv('Placement_Data_Full_Class.csv')

print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)

print('Statistical information of Numerical Columns: \n',df.describe())

print('Total Number of Null Values in Dataset:', df.isna().sum())

df['sl_no']=df['sl_no'].astype('int8')
print('Check Datatype of sl_no ',df.dtypes)
df['ssc_p']=df['ssc_p'].astype('int8')
print('Check Datatype of ssc_p ',df.dtypes)

df=pd.read_csv('Placement_Data_Full_Class.csv')
print('Placement dataset is successfully loaded into DataFrame ...... ')
print(df.head().T)
df['gender'].replace(['M','F'],[0,1],inplace=True)
print('\nFind Male and replace it by 0 and Find Female and replace it by 1:\n',df['gender'].head(10))
# print()

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
df[['gender']]=enc.fit_transform(df[['gender']])
print(df.head().T)

df=pd.read_csv('Placement_Data_Full_Class.csv')
print('Placement dataset is successfully loaded into DataFrame ...... ')
print(df.head().T)

df['salary'] =df['salary']/df['salary'].abs().max()
print(df.head().T)

from sklearn.preprocessing import MaxAbsScaler
abs_scaler=MaxAbsScaler()
df[['salary']]=abs_scaler.fit_transform(df[['salary']])
print('\n Maximum absolute Scaling method normalization -1 to 1 \n\n')
print(df.head())

df=pd.read_csv('Placement_Data_Full_Class.csv')
print('Placement dataset is successfully loaded into DataFrame ...... ')
print(df.head().T)
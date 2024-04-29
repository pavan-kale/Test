import pandas as pd 
df = pd.read_csv('titanic.csv') 
print('\033[1m titanic Dataset is successfully loaded ....\033[0m\n') 
 
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 
 
       
choice = 1 
while(choice != 10): 
 print('------------------- Menu ----------------------------') 
 print('1.Display information of dataset') 
 print('2.Find Missing values') 
 print('3.Fill Missing values') 
 print('4.Histogram of 1-variables (Age & Fare)') 
 print('5.Histogram of 2-variables') 
 print('10. Exit') 
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
  print('Total Number of Null Values in Dataset:', df.isna().sum()) 
 
 if choice == 3: 
 
  df['Age'] =df['Age'].fillna(df['Age'].median()) 
  print('Null values are: \n',df.isna().sum()) 
 
 if choice == 4: 
 
  # fig= plt.figure(figsize=(10,7)) 
  # sns.boxplot(data = df, x ='Age') 
  # plt.title('Boxplot with 1 variable i.e. Age') 
  # plt.show() 
  fig, axes = plt.subplots(1,2) 
  fig.suptitle('Histogram 1-variables (Age & Fare)') 
  sns.histplot(data = df, x ='Age', ax=axes[0]) 
  sns.histplot(data = df, x ='Fare', ax=axes[1]) 
  plt.show() 
 
 if choice == 5: 
 
  fig, axes = plt.subplots(2,2) 
  fig.suptitle('Histogram of 2-variables') 
  sns.histplot(data = df, x ='Age',hue='Survived', multiple='dodge', ax=axes[0,0]) 
  sns.histplot(data = df, x ='Fare',hue='Survived', multiple='dodge', ax=axes[0,1]) 
  sns.histplot(data = df, x ='Age',hue='Sex', multiple='dodge', ax=axes[1,0]) 
  sns.histplot(data = df, x ='Fare',hue='Sex', multiple='dodge', ax=axes[1,1]) 
  plt.show()
import pandas as pd 
import numpy as np 
df = pd.read_csv('Placement_Data_Full_Class.csv') 
choice = 1 
while(choice != 10): 
 print('----------- Menu -------------') 
 print('1.Display information of dataset') 
 print('2.Display Statistical information of Numerical Columns') 
 print('3.Find Missing values') 
 print('4.Change Datatype of Columns') 
 print('5.Conversion of categorical to quantitative') 
 print('6.Nomralization using Min-Max scaling') 
 print('7.Load dataset') 
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
  print('Statistical information of Numerical Columns: \n',df.describe()) 
 
 if choice == 3: 
  print('Total Number of Null Values in Dataset:', df.isna().sum()) 
 
 if choice == 4: 
  df['sl_no']=df['sl_no'].astype('int8') 
  print('Check Datatype of sl_no ',df.dtypes) 
  df['ssc_p']=df['ssc_p'].astype('int8') 
  print('Check Datatype of ssc_p ',df.dtypes) 
 
 if choice == 5:   
  choice1 = 'a' 
  while(choice1 != 'e'): 
 
   print('a. Find And Replace Method') 
   print('b. Using Scikit-Learn Library') 
   print('c. Go back') 
   choice1 = input('Enter your choice for conversion of Categorical to Quantitative: ') 
   df=pd.read_csv('Placement_Data_Full_Class.csv') 
   print('Placement dataset is successfully loaded into DataFrame ...... ') 
   print(df.head().T) 
 
   if choice1 == 'a': 
    # df['gender'].replace(['M','F'],[0,1],inplace=True) 
    df.replace({'gender' : {'M':0, 'F':1}}, inplace=True)
    print('\nFind Male and replace it by 0 and Find Female and replace it by 1:\n',df['gender'].head(10)) 
    # print() 
 
   if choice1 == 'b': 
    from sklearn.preprocessing import OrdinalEncoder 
    enc = OrdinalEncoder() 
    df[['gender']]=enc.fit_transform(df[['gender']]) 
    print(df.head().T) 
     
   if choice1 == 'c': 
    break 
 if choice == 6: 
  choice2 = 'a' 
  while(choice2 != 'f'): 
   print('a. Maximum Absolute Scaling') 
   print('b. Using Sci-kit learn') 
   print('c. Go back') 
   choice2 = input('Enter your choice: ') 
 
   df=pd.read_csv('Placement_Data_Full_Class.csv') 
   print('Placement dataset is successfully loaded into DataFrame ...... ') 
   print(df.head().T) 
 
   if choice2 == 'a': 
    df['salary'] =df['salary']/df['salary'].abs().max() 
    print(df.head().T) 
   if choice2 == 'b': 
    from sklearn.preprocessing import MaxAbsScaler 
    abs_scaler=MaxAbsScaler() 
    df[['salary']]=abs_scaler.fit_transform(df[['salary']]) 
    print('\n Maximum absolute Scaling method normalization -1 to 1 \n\n') 
    print(df.head()) 
   if choice2 == 'c': 
    break 
 
 if choice == 7: 
  df=pd.read_csv('Placement_Data_Full_Class.csv') 
  print('Placement dataset is successfully loaded into DataFrame ...... ') 
  print(df.head().T) 
     
 if choice == 10: 
  break

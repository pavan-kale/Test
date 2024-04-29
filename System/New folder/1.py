import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

df = pd.read_csv('Placement_Data_Full_Class.csv')
print(df.head(10))
print("it gives how many rows and how many columns")
print(df.shape) 
print("This displays names of columns with their type")
print(df.columns) 
print(df.size) 

print('\nData types of gender=', df.dtypes['gender'],'\nData types of sl_no=', df.dtypes['sl_no'])
A={'gender': 'string', 'sl_no':'int32'}
df=df.astype(A)
print('\nData types of gender after label encoding = ', df.dtypes['gender'],'\nData types of sl_no after label encoding = ', df.dtypes['sl_no'],'\n')

df['status'].replace(['Placed', 'Not Placed'],[0, 1])
print(df.head(10))
df["status"] = df["status"].astype('category')
print('This is encoding function')
df["status"] = df["status"].cat.codes
print(df.head(10))
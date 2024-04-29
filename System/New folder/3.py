import pandas as pd
df = pd.read_csv('Employee_Salary.csv')
print(df)

print('print all rows and columns')
print(df.shape)

print('data types in our set')
print(df.dtypes)

print('max salary', df['Salary'].max())

print('mean of the salary', df['Salary'].mean())

print('median of the salary', df['Salary'].median())

print('std of the salary', df['Salary'].std())

print('min age is', df['Age'].min())

print('max age is:', df['Age'].max())

smallest_number= df['Age'].min()
print(smallest_number)




df1 = pd.read_csv('iris.csv')
print(df1)

print('print all rows and columns')
print(df1.shape)

print('data types in our set')
print(df1.dtypes)

print('max Sepal_Length', df1['Sepal_Length'].max())
print('max petal_length', df1['Petal_Length'].max())

print('mean of the Sepal_Length', df1['Sepal_Length'].mean())
print('mean of the petal_length', df1['Petal_Length'].mean())

print('median of the Sepal_Length', df1['Sepal_Length'].median())
print('median of the petal_length', df1['Petal_Length'].median())

print('std of the Sepal_Length', df1['Sepal_Length'].std())
print('std of the petal_length', df1['Petal_Length'].std())

print('min sepal length is', df1['Sepal_Length'].min())
print('min petal_length is', df1['Petal_Length'].min())

print('max sepal length is:', df1['Sepal_Length'].max())
print('max petal_length is:', df1['Petal_Length'].max())



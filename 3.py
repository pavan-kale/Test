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

print('max sepal_length', df1['sepal_length'].max())
print('max petal_length', df1['petal_length'].max())

print('mean of the sepal_length', df1['sepal_length'].mean())
print('mean of the petal_length', df1['petal_length'].mean())

print('median of the sepal_length', df1['sepal_length'].median())
print('median of the petal_length', df1['petal_length'].median())

print('std of the sepal_length', df1['sepal_length'].std())
print('std of the petal_length', df1['petal_length'].std())

print('min sepal length is', df1['sepal_length'].min())
print('min petal_length is', df1['petal_length'].min())

print('max sepal length is:', df1['sepal_length'].max())
print('max petal_length is:', df1['petal_length'].max())



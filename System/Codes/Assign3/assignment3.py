import seaborn as sns 
import pandas as pd

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Display the first few rows of the dataset 
print("First few rows of the Iris dataset:")
print(iris.head())

# 1. Display basic statistical details for 'Iris-setosa'
setosa_stats = iris[iris['species'] == 'setosa'].describe()

# Display the statistical details for 'Iris-setosa'
print("\nStatistical details for 'Iris-setosa':")
print(setosa_stats)

# 2. Display basic statistical details for 'Iris-versicolor'
versicolor_stats = iris[iris['species'] == 'versicolor'].describe()

# Display the statistical details for 'Iris-versicolor'
print("\nStatistical details for 'Iris-versicolor':")
print(versicolor_stats)

# 3. Display basic statistical details for 'Iris-virginica'
virginica_stats = iris[iris['species'] == 'virginica'].describe()

# Display the statistical details for 'Iris-virginica'
print("\nStatistical details for 'Iris-virginica':")
print(virginica_stats)
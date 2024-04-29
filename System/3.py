# Import necessary libraries 
import pandas as pd 
 
# Load the Titanic dataset (you can replace this with your dataset) titanic_df = 
df = pd.read_csv('titanic.csv') 
 
# Display first few rows of the dataset print("First few rows of the Titanic dataset:") 
print(df.head()) 
 
# 1. Provide summary statistics grouped by a categorical variable 
# Let's use the 'Pclass' (passenger class) as the categorical variable and 'Age' as the quantitative variable 
grouped_stats = df.groupby('Pclass')['Age'].describe() 
 
# Display the summary statistics 
print("\nSummary statistics of Age grouped by Pclass:") 
print(grouped_stats) 
 
# 2. Create a list that contains a numeric value for each response to the categorical variable 
# In this case, create a list of mean ages for each passenger class 
mean_age_by_class = df.groupby('Pclass')['Age'].mean().tolist() 
 
# Display the list of mean ages for each passenger class 
print("\nMean Age for each Passenger Class:") 
print(mean_age_by_class) 
import pandas as pd
df = pd.read_csv('titanic.csv')
print('\033[1m titanic Dataset is successfully loaded ....\033[0m\n')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df['Age'].fillna(df['Age'].median(), inplace = True)
print('Null values are: \n',df.isna().sum())

sns.histplot(data = df,x='Fare', hue='Sex',multiple='dodge')
plt.title('Histogram with 2 variable i.e. Age and Sex')
plt.show()

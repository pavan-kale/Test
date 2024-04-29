def DetectOutlier(df, var): 
 Q1 = df[var].quantile(0.25) 
 Q3 = df[var].quantile(0.75) 
 IQR = Q3 - Q1 
 high, low = Q3+1.5*IQR, Q1-1.5*IQR 
 
 print("Highest allowed in variable:", var, high)  
 print("lowest allowed in variable:", var, low) 
 count = df[(df[var] > high) | (df[var] < low)][var].count() 
 print('Total outliers in:', var,':',count) 
 
 df = df[((df[var] >= low) & (df[var] <= high))] 
 print('Outliers removed in', var) 
 return df 
 
def DrawBoxplot(df, msg): 
 fig, axes = plt.subplots(1,2) 
 fig.suptitle(msg) 
 sns.boxplot(data = df, x ='RM', ax=axes[0]) 
 sns.boxplot(data = df, x ='LSTAT', ax=axes[1]) 
 fig.tight_layout() 
 plt.show() 
 
#read dataset 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
 
df = pd.read_csv('Boston.csv') 
print('Boston dataset is succesfully loaded ..............................') 
choice = 1 
while (choice != 10): 
 print('---------------Menu-----------------') 
 print('1. Display') 
 print('2. Find Missing values') 
 print('3. Detect and remove outliers') 
 print('4. Encoding using label encoder') 
 print('5. Find correlation matrix') 
 print('6. Train and Test the model using Linear Regration Model') 
 print('7. Predict House Price by giving user Input') 
 print('8. Reloaded Boston Housing dataset.....') 
 print('10. Exit') 
 choice = int(input('Enter your choice: ')) 
 
 if choice == 1: 
  print(df.head().T) 
  print(df.columns)  
 
 if choice == 2: 
  print(df.isnull().sum()) 
 
 if choice == 3: 
  Column_Name = ['LSTAT','RM'] 
  Output = ['MEDV'] 
 
  DrawBoxplot(df, 'Before removing Outliers') 
  print('Identifying overall outliers in Column Name variables.....') 
  for var in Column_Name: 
   df = DetectOutlier(df,var) 
 
  df = DetectOutlier(df, 'RM') 
  DrawBoxplot(df, 'After removing Outliers') 
 
 if choice == 4: 
  df['MEDV']=df['MEDV'].astype('category') 
  print(df.dtypes) 
  df['MEDV']=df['MEDV'].cat.codes 
  print(df) 
 
  print(df.isnull().sum()) 
 
 if choice == 5: 
  import seaborn as sns 
  import matplotlib.pyplot as plt  
  sns.heatmap(df.corr(),annot=True) 
  plt.show() 
 
 if choice == 6: 
 
  x = df[['RM','LSTAT']] 
  y = df['MEDV'] 
  from sklearn.model_selection import train_test_split 
  x_train, x_test, y_train, y_test =train_test_split(x, y, test_size= 0.20, random_state = 0) 
  print('x_train = ',x_train) 
  print('x_test = ', x_test) 
  from sklearn.linear_model import LinearRegression  
  model=LinearRegression().fit(x_train,y_train) 
  y_pred=model.predict(x_test) 
  print('y_pred :' ,y_pred) 
  print('y_test :' ,y_test) 
  from sklearn.metrics import mean_absolute_error 
  print('MAE: ',mean_absolute_error(y_test,y_pred)) 
  print("Model Score: ", model.score(x_test,y_test)) 
 
 if choice == 7: 
  import numpy as np  
  features = np.array([[6,19]]) 
  prediction = model.predict(features) 
  print('Prediction: {}'.format(prediction)) 
 
 if choice == 8: 
  df=pd.read_csv('Boston.csv') 
  print("Successfully Reloaded the Boston Housing Dataset*********") 
  print("Boston Housing Dataset \n",df) 
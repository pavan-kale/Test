# check for outliers by using IQR method
# co-relation matrix
def DetectOutlier(df,var):

    
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3+1.5*IQR, Q1-1.5*IQR
    
    print("Highest allowed in variable:", var, high)
    print("lowest allowed in variable:", var, low)
    count = df[(df[var] > high) | (df[var] < low)][var].count()
    print('Total outliers in:',var,':',count)

    df = df[((df[var] >= low) & (df[var] <= high))]
    print('Outliers removed in', var)
    return df

def Display(y_pred, y_test):

    from sklearn.metrics import confusion_matrix

    # passing actual and predicted values
    cm = confusion_matrix(y_test, y_pred)
    print('confusion_matrix\n',cm)
    # true write data values in each cell of the matrix
    sns.heatmap(cm, annot=True)
    plt.show()

    # importing classification report
    from sklearn.metrics import classification_report

    # printing the report
    print(classification_report(y_test, y_pred))



# Read Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
print('Iris dataset is successfully loaded .....')
choice = 1
while(choice != 10):
    print('-------------- Menu --------------------')
    print('1. Display information of dataset')
    print('2. Find Missing values')
    print('3. Detect and remove outliers')
    print('4. Encoding using label encoder')
    print('5. Find correlation matrix')
    print('6. Train and Test the model using Bernoulli Naive Bayes and Display Confusion matrix and Accuracy Score')
    print('7. Train and Test the model using Gaussian Naive Bayes and Display Confusion matrix and Accuracy Score')
    print('8. Prediction by using User Input ')
    print('10. Exit')
    choice = int(input('Enter your choice: '))

    if choice == 1:
        print(df.head())
        print(df.columns)
        print(df['variety'].unique())

    if choice == 2:
        print(df.isnull().sum())

    if choice == 3:
        variety = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
        species = ['Setosa','Versicolor','Virginica']

        fig, axes = plt.subplots(2,2)
        fig.suptitle('Before removing Outliers')
        sns.boxplot(data = df, x ='sepal.length', ax=axes[0,0])
        sns.boxplot(data = df, x ='sepal.width', ax=axes[0,1])
        sns.boxplot(data = df, x ='petal.length', ax=axes[1,0])
        sns.boxplot(data = df, x ='petal.width', ax=axes[1,1])
        plt.show()

        print('Identifying overall outliers in feature variables.....')
        for var in variety:
            df = DetectOutlier(df,var)

        fig, axes = plt.subplots(2,2)
        fig.suptitle('After removing Outliers')
        sns.boxplot(data = df, x ='sepal.length', ax=axes[0,0])
        sns.boxplot(data = df, x ='sepal.width', ax=axes[0,1])
        sns.boxplot(data = df, x ='petal.length', ax=axes[1,0])
        sns.boxplot(data = df, x ='petal.width', ax=axes[1,1])
        fig.tight_layout()
        plt.show()

    if choice == 4:
        df['variety']=df['variety'].astype('category')
        print(df.dtypes)
        df['variety']=df['variety'].cat.codes
        print(df)

        print(df.isnull().sum())

    if choice == 5:
        # import plotly.express as px
        # c1=df.corr()
        # fig = px.imshow(c1,text_auto=True)
        # fig.show()

        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.heatmap(df.corr(),annot=True)
        plt.show()

    if choice == 6:

        # Split the data into inputs and outputs
        X = df.iloc[:, [0,1,2,3]].values
        y = df.iloc[:, 4].values

        # Training and testing data
        from sklearn.model_selection import train_test_split
        
        # Assign test data size 25%
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size= 0.25, random_state=0)

            # Importing standard scaler
        from sklearn.preprocessing import StandardScaler
        # Scalling the input data
        sc_X = StandardScaler() 
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.fit_transform(X_test)
        
        # importing classifier
        print('X_train=',X_train)
        print('X_test=',X_test)
          
        from sklearn.naive_bayes import BernoulliNB

        # initializaing the NB
        classifer = BernoulliNB()

        # training the model
        classifer.fit(X_train, y_train)

        # testing the model
        y_pred = classifer.predict(X_test)
        print('y_pred=',y_pred)
        # importing accuracy score
        from sklearn.metrics import accuracy_score

        print('Accuracy of the BernoulliNB model : ')
        print(accuracy_score(y_pred, y_test))
        Display(y_pred, y_test)

    if choice == 7:
        # import Gaussian Naive Bayes classifier
        from sklearn.naive_bayes import GaussianNB

        # create a Gaussian Classifier
        classifer1 = GaussianNB()

        # training the model
        classifer1.fit(X_train, y_train)

        # testing the model
        y_pred1 = classifer1.predict(X_test)
        print('y_pred1=',y_pred1)
        # importing accuracy score
        from sklearn.metrics import accuracy_score

        print('printing the accuracy of the GaussianNB model=')
        print(accuracy_score(y_test,y_pred1))
        Display(y_pred1, y_test)
    
    if choice == 8:
        import numpy as np
        features = np.array([[5,2.9,1,0.2]])
        prediction = classifer.predict(features)
        print('Prediction: {}'.format(prediction))
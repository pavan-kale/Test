import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
# Load the dataset
df = pd.read_csv('iris.csv')
# Replace species names with numeric labels
df['Species'].replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], inplace=True)
# Split data into features (X) and target (Y)
X = df.iloc[:, 1:5]
Y = df['Species']
# Split data into training and validation sets
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.33, 
random_state=42)
# Initialize Gaussian Naive Bayes classifier
clf = GaussianNB()
# Fit the model on the training set
clf.fit(X_train, Y_train)
# Predictions on the validation set
Y_pred = clf.predict(X_validation)
# Compute confusion matrix
cm = confusion_matrix(Y_validation, Y_pred)
# Extract TP, FP, TN, FN
TP = cm[1, 1]
FP = cm[0, 1]
TN = cm[0, 0]
FN = cm[1, 0]
# Compute other metrics
accuracy = accuracy_score(Y_validation, Y_pred)
error_rate = 1 - accuracy
precision = precision_score(Y_validation, Y_pred, average='weighted')
recall = recall_score(Y_validation, Y_pred, average='weighted')
print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initializing the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Training the model
clf.fit(X_train, y_train)

# Making predictions on the test data
y_pred = clf.predict(X_test)

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)

# Displaying results
print(f'Predicted labels: {y_pred}')
print(f'Actual labels: {y_test}')
print(f'Accuracy: {accuracy:.2f}')

import pandas as pd
import matplotlib.pyplot as plt
import dataProcessor as data
import numpy as np



dataset = data.AvocadoCleaner().process_data()
X = dataset.iloc[:, 1].values

dataset_result = data.AvocadoCleaner().process_data()
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
print(X_train)
print(y_train)
# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test.reshape(-1, 1)[:200])

# Visualising the Training set results
plt.scatter(X_train[:200], y_train[:200], color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Avocado Price vs Total Volume (Training set)')
plt.xlabel('Volume')
plt.ylabel('Price')
plt.show()

# Visualising the Test set results
plt.scatter(X_test[:200], y_test[:200], color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Avocado Price vs Total Volume (Training set)')
plt.xlabel('Volume')
plt.ylabel('Price')
plt.show()

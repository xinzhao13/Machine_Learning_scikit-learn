# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0]
y = dataset.iloc[:, 1]


# Split data into train- and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Convert from type series object of pandas.core.series module to float64
#   because 'Series' object of pandas.core.series module has no attribute 'reshape'
X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values


# Reshape prior to fiting linear regression
#X_train = X_train.reshape(1, -1)
#y_train = y_train.reshape(1, -1)
#X_test = X_test.reshape(1, -1)
#y_test = y_test.reshape(1, -1)


# Initialize and fit simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train)


# Predict test set results
y_predictions = regressor.predict(X_test.reshape(-1, 1))


# Visualize train results
"""plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train.reshape(-1, 1)), color = 'green')
plt.title('Salary : Experience (Train)')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary (USD)')
plt.show()"""


# Visualize test results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train.reshape(-1, 1)), color = 'green')
plt.title('Salary : Experience (Test)')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary (USD)')
plt.show()
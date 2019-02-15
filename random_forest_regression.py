# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Initialize and fit the ensemble learning
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 337)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(np.array(6.5).reshape(1, -1))

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

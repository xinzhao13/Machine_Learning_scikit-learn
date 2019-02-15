# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, 4]


# Encode categorical data
X = X.values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [3]) #categorical-features keyboard is deprecated; use ColumnTransformer instead in v0.22+
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X = onehotencoder.fit_transform(X).toarray()


# Adressing dummy variable trap (not necessary as the library handles this anyway)
X = X[:, 1:]


# Feature scaling (not needed as the library does this anyway)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X[:, 2:5] = sc_X.fit_transform(X[:, 2:5])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Initialize and fit multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Test set results prediction
y_predictions = regressor.predict(X_test)


# Add bias as it is not accounted for by statsmodel library
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)


# Optimize model with backward elimination
#   (manual process; p-value = 0.050)
import statsmodels.formula.api as sm

X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()
# remove x2 as P>|t| = 0.990

X_optimal = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()
# remove x1 as P>|t| = 0.940

X_optimal = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()
# remove x2 as P>|t| = 0.602

X_optimal = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()
# remove x2 as P>|t| = 0.060.

X_optimal = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()
# model optimized

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
X[:, 3:6] = sc_X.fit_transform(X[:, 3:6])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Logistic Regression


# 0 - Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# 1 - Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [1, 2, 3]]
y = dataset.iloc[:, 4].values



# 2 - Encode the independent variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X = X.values
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])




# 3 - Splitting the dataset into the training- and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)



# 4 - Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



# 5 - Fitting the model to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)



# 6 - Predicting the test set results
y_predictions = classifier.predict(X_test)
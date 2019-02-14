# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 3]


# Adressing missing data
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


# Confirming missing data adressed
x = pd.DataFrame(x)
y = pd.DataFrame(y)
# Needs to be done this way as Spyder IDE 3.3.2 has issues with showing arrays
#   that contain different types of objects. May be fixed in future versions (?)
#   [step can be possibly disregarded in other IDEs]


# Encoding categorical data
x = x.values
y = y.values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [0])

# Encoding the independent variables
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
x = onehotencoder.fit_transform(x).toarray()

# Encoding the dependent variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.20)


# Feature scaling implementation
from sklearn.model_selection import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #No need to fit as μ and σ are already calculated from train set










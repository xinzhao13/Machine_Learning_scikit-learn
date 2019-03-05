# Logistic Regression


# 0 - Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# 1 - Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]]
y = dataset.iloc[:, 4].values



# 2 - Encode the independent variable (removed as I removed the gender variable for better results)
#from sklearn.preprocessing import LabelEncoder
#labelencoder_X = LabelEncoder()
#X = X.values
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])



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



# 6 - Predicting and visualizing results
y_predictions = classifier.predict(X_test)

# 6.1 - Making the confusion matrix
from sklearn.metrics import confusion_matrix
confm = confusion_matrix(y_test, y_predictions)

# 6.2 - Visualizing the test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

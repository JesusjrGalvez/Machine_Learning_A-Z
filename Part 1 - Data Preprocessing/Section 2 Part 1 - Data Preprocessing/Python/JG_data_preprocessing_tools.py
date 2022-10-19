import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection

sklearn.preprocessing
sklearn.compose
sklearn.model_selection
sklearn.impute
sklearn.model_selection.train_test_split()

# Get dataset from file
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Impute missing values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# One hot encoding of categorical values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# below, the [0] corresponds with the index of the column where you want to apply one hot enconding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],remainder='passthrough' )
X = np.array(ct.fit_transform(X))

# One hot encoding of dependent variables

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

# Split dataset into training and test sets

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
# print(X_train)
# print('----')
# print(X_test)
# print('llll')
# print(Y_train)
# print('----')
# print(Y_test)

# Standarization of values


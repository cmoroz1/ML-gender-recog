import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('voice.csv')
X = data.values[:,0:-1]
Y = data.values[:,-1]

# Checks that all values in the data matrix are numbers
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if(not isinstance(X[i,j], numbers.Number)):
            print("NAN @ %d %d" % (i,j))

# Converts 'male' -> 1 and 'female' -> -1
for i in range(Y.shape[0]):
    if(not isinstance(Y[i], numbers.Number)):
        if(Y[i] == 'male'):
            Y[i] = 1
        elif(Y[i] == 'female'):
            Y[i] = -1
        else:
            print('not found')

# Splits the data into a training set and testing set
# If not specified, testing is 25% of all the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# Scale all training features to have a mean of zero and std dev of 1
# Will only use the training data to determine how to scale the features
# Testing data will have a mean of approx 0 and a std dev of approx 1
scaler = StandardScaler().fit(X_train)
norm_X_train = scaler.transform(X_train)
norm_X_test = scaler.transform(X_test)

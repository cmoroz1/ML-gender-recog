import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('voice.csv')

# Converts 'male' -> 1 and 'female' -> -1
data = data.replace(['male','female'],[1,-1])
X = data.values[:,0:-1]
Y = data.values[:,-1]

# Checks that all values in the data matrix are numbers
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if(not isinstance(X[i,j], numbers.Number)):
            print("NAN @ %d %d" % (i,j))

# Splits the data into a training set and testing set
# If not specified, testing is 25% of all the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

# Scale all training features to have a mean of zero and std dev of 1
# Will only use the training data to determine how to scale the features
# Testing data will have a mean of approx 0 and a std dev of approx 1
scaler = StandardScaler().fit(X_train)
norm_X_train = scaler.transform(X_train)
norm_X_test = scaler.transform(X_test)

# Create MLP Classifier object, with set max_iter and random_state (was 80)
mlp = MLPClassifier(max_iter=80, random_state=0)
mlp.fit(norm_X_train,Y_train)

# Visualize the accuracy of the model
print("Training set score: %f" % mlp.score(norm_X_train, Y_train))
print("Test set score: %f" % mlp.score(norm_X_test, Y_test))

################################################################################
# Used to find the optimal parameters for the model
################################################################################
# mlp = MLPClassifier(random_state=0)
# alphas = 10.0 ** -np.arange(1,7)
# iterations = np.arange(20,2000,20)
# params = {'alpha':alphas, 'max_iter':iterations}
# clf = GridSearchCV(mlp, params)
# clf.fit(norm_X_train,Y_train)
# print(clf.best_params_)

################################################################################
# Used to visualize how the accuracy changed with max_iter
################################################################################
# train_scores = []
# test_scores = []
# for i in np.arange(70,90,1):
#     print("Training %d max_iter" % (i))
#     mlp = MLPClassifier(max_iter=i, random_state=0)
#     mlp.fit(norm_X_train,Y_train)
#     train_scores.append(mlp.score(norm_X_train, Y_train))
#     test_scores.append(mlp.score(norm_X_test, Y_test))
#
# plt.plot(np.arange(70,90,1), train_scores)
# plt.plot(np.arange(70,90,1), test_scores)
# plt.legend(['train acc','test acc'])
# plt.show()

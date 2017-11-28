import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support

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

# Create MLP Classifier object, with random_state = 0
mlp = MLPClassifier(max_iter = 500, random_state=0)
mlp.fit(norm_X_train,Y_train)
mlp_Y_pred = mlp.predict(norm_X_test)
mlp_metrics = precision_recall_fscore_support(mlp_Y_pred, Y_test)
mlp_precision = np.average(mlp_metrics[0]) # First array is precision of each class
mlp_recall = np.average(mlp_metrics[1]) # Second array is accuracy of each class
mlp_f1 = 2 * (mlp_precision * mlp_recall) / (mlp_precision + mlp_recall)

# Create a SVM classifier with optimal parameters specified below
svm = SVC(C = 2, kernel = 'rbf')
svm.fit(norm_X_train,Y_train)
svm_Y_pred = svm.predict(norm_X_test)
svm_metrics = precision_recall_fscore_support(svm_Y_pred, Y_test)
svm_precision = np.average(svm_metrics[0]) # First array is precision of each class
svm_recall = np.average(svm_metrics[1]) # Second array is accuracy of each class
svm_f1 = 2 * (svm_precision * svm_recall) / (svm_precision + svm_recall)

# Output the metrics of the models
print("Neural Network Model Test Set Metrics:")
print("\tAccuracy:\t%f" % mlp.score(norm_X_test, Y_test))
print("\tPrecision:\t%f" % mlp_precision)
print("\tRecall:\t\t%f" % mlp_recall)
print("\tF1:\t\t%f" % mlp_f1)
print("")
print("Support Vector Machine Model Test Set Metrics:")
print("\tAccuracy:\t%f" % svm.score(norm_X_test, Y_test))
print("\tPrecision:\t%f" % svm_precision)
print("\tRecall:\t\t%f" % svm_recall)
print("\tF1:\t\t%f" % svm_f1)

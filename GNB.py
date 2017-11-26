from sklearn import datasets
from Preprocessing import Preprocess
import numpy as np

# load iris dataset for testing
from sklearn.model_selection import train_test_split

creditdata = Preprocess("default of credit card clients.xls")
data_X, data_Y = creditdata.load_dataset()
X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2)

# Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
C_G = gnb.fit(X_train, Y_train)
y_GNBpred = C_G.predict(X_test)
print("Number of mislabeled points(Gaussian NB) out of a total %d points : %d" % (X_test.shape[0],(Y_test !=
                                                                                          y_GNBpred).sum()))
Precision = 1 - (Y_test != y_GNBpred).sum()/X_test.shape[0]
print("Precision of Gaussian NB is %4.2f" %Precision)

# K Nearest Neighbors Classifier (plot)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
C_K = knn.fit(X_train, Y_train)
y_KNNpred = C_K.predict(X_test)
print("Number of mislabeled points(K Nearest Neighbors) out of a total %d points : %d" % (X_test.shape[0],(Y_test !=
                                                                                          y_KNNpred).sum()))
Precision = 1 - (Y_test != y_KNNpred).sum()/X_test.shape[0]
print("Precision of KNN is %4.2f" %Precision)

# Support Vector Classifier (plot)
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, Y_train)
y_SVMpred = clf.predict(X_test)
print("Number of mislabeled points(SVM) out of a total %d points : %d" % (X_test.shape[0],(Y_test !=
                                                                                            y_SVMpred).sum()))
Precision = 1 - (Y_test != y_SVMpred).sum()/X_test.shape[0]
print("Precision of SVM is %4.2f" %Precision)
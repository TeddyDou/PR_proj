from sklearn import datasets
from Preprocessing import Preprocess
import numpy as np
from sklearn import preprocessing

# load iris dataset for testing
from sklearn.model_selection import train_test_split

creditdata = Preprocess("default of credit card clients.xls")
# data_X, data_Y = creditdata.load_dataset()
X_train, X_test, Y_train, Y_test = creditdata.load_dataset()


# X_train, X_test, Y_train, Y_test = train_test_split(data_X, data_Y, test_size=0.2)
#1.StandardScaler
# scaler =  preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train) 
# X_test = scaler.transform(X_test)  

#2.MinMaxScaler
# scaler = preprocessing.MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

#3.MaxAbsScaler
# scaler = preprocessing.MaxAbsScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

#4.Normalization low
# normalizer = preprocessing.Normalizer().fit(X_train)
# X_train = normalizer.transform(X_train)     
# X_test = normalizer.transform(X_test)   

#5.Binarization
# binarizer = preprocessing.Binarizer().fit(X_train)
# X_train = binarizer.transform(X_train)     
# X_test = binarizer.transform(X_test)   

#6. Non-linear transformation¶
# quantile_transformer = preprocessing.QuantileTransformer( output_distribution='normal', random_state=0)
# X_train = quantile_transformer.fit_transform(X_train)
# X_test = quantile_transformer.transform(X_test)
# print(X_train)


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
from sklearn import datasets
import numpy as np

# load iris dataset for testing
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2)

print(iris_X)
print(iris_y)

# Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
C_G = gnb.fit(X_train, y_train)
y_GNBpred = C_G.predict(X_test)
print("Number of mislabeled points(Gaussian NB) out of a total %d points : %d" % (X_test.shape[0],(y_test !=
                                                                                          y_GNBpred).sum()))

# Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
C_M = mnb.fit(X_train, y_train)
y_MNBpred = C_M.predict(X_test)
print("Number of mislabeled points(Multinomial NB) out of a total %d points : %d" % (X_test.shape[0],(y_test !=
                                                                                          y_MNBpred).sum()))

# K Nearest Neighbors Classifier (plot)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
C_K = knn.fit(X_train, y_train)
y_KNNpred = C_K.predict(X_test)
print("Number of mislabeled points(K Nearest Neighbors) out of a total %d points : %d" % (X_test.shape[0],(y_test !=
                                                                                          y_KNNpred).sum()))

# Support Vector Classifier (plot)
from sklearn import svm
C = 1
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
C_SVM = (clf.fit(X_train, y_train) for clf in models)
y_SVMpred = (clf.predict(X_test) for clf in C_SVM)
for pred in y_SVMpred:
    print("Number of mislabeled points(SVM) out of a total %d points : %d" % (X_test.shape[0], (y_test !=pred).sum()))
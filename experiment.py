"""
Created on Nov.25, 2017

@author Ted
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, Binarizer, QuantileTransformer
from Preprocessing import Preprocess


class experiment:
    def __init__(self):
        self.classifier = []
        self.processor =[]
        self.result = []
        creditdata = Preprocess("default of credit card clients.xls")
        self.X_train, self.X_test, self.Y_train, self.Y_test = creditdata.load_dataset()
        self.buildclf()
        self.buildprocessor()

        # # code for testing quick access dataset
        # import pickle
        # with open('express_x', 'rb') as fp:
        #     self.X_train = pickle.load(fp)
        #     self.X_test = pickle.load(fp)
        #     fp.close()
        # with open('express_y', 'rb') as fp:
        #     self.Y_train = pickle.load(fp)
        #     self.Y_test = pickle.load(fp)
        #     fp.close()

    def buildclf(self):
        self.classifier.append(processor(GaussianNB(), "Gaussian NB"))
        self.classifier.append(processor(KNeighborsClassifier(n_neighbors=15), "K Nearest Neighbors"))
        self.classifier.append(processor(SVC(), "C-Support Vector"))
        self.classifier.append(processor(LogisticRegression(), "Logistic Regression"))
        self.classifier.append(processor(LinearDiscriminantAnalysis(), "Linear Discriminant Analysis"))
        self.classifier.append(processor(MLPClassifier(), "Artificial neural networks"))
        self.classifier.append(processor(DecisionTreeClassifier(), "Decision Tree"))

    def buildprocessor(self):
        self.processor.append(processor(StandardScaler(), "Standard Scaler"))
        self.processor.append(processor(MinMaxScaler(), "MinMax Scaler"))
        self.processor.append(processor(MaxAbsScaler(), "MaxAbs Scaler"))
        self.processor.append(processor(Normalizer(), "Normalization"))
        self.processor.append(processor(Binarizer(), "Binarization"))
        self.processor.append(processor(QuantileTransformer(), "Non-linear transformation"))

    def getprecision(self, clf, x1, x2, y1, y2):
        clf.obj.fit(x1, y1)
        y_pred = clf.obj.predict(x2)
        mislabeled = (y2 != y_pred).sum()
        totaltest = x2.shape[0]
        print("Mislabeled points (%s Classification) out of a total %d points : %d" % (clf.descr, totaltest,
                                                                                       mislabeled))
        Precision = 1 - mislabeled / totaltest
        print("Precision of %s is %4.2f%%" % (clf.descr, Precision * 100))
        return Precision

    def run(self):

        for clf in self.classifier:
            row = []
            prec = self.getprecision(clf, self.X_train, self.X_test, self.Y_train, self.Y_test)
            row.append(prec)
            for processor in self.processor:
                processed_X_train = processor.obj.fit_transform(self.X_train)
                processed_X_test = processor.obj.fit_transform(self.X_test)
                prec = self.getprecision(clf, processed_X_train, processed_X_test, self.Y_train, self.Y_test)
                row.append(prec)
            self.result.append(row)
        self.printresult()

    def printresult(self):
        print("%28s%28s" % ("___________________________|", "No preprocess"), end='')
        for p in self.processor:
            print("%28s" %p.descr, end='')
        print('')
        for i in range(len(self.classifier)):
            print("%28s" % self.classifier[i].descr, end='')
            for j in range(len(self.processor)+1):
                print("%22s%4.2f%%" % ("", self.result[i][j]*100), end='')
            print('')


class processor:
    def __init__(self, obj, str):
        self.obj = obj
        self.descr = str


if __name__ == '__main__':
    a = experiment()
    a.run()


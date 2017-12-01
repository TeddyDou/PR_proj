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
from PreprocessingAlt import PreprocessAlt
from PostprocessingAlt import PostprocessAlt


class experiment:
    def __init__(self):
        self.classifier = []
        self.processor =[]
        self.result = []
        creditdata = PreprocessAlt("default of credit card clients.xls")
        self.raw_X_train, self.raw_X_test, self.raw_Y_train, self.raw_Y_test = creditdata.load()
        self.low_dim_X_train, self.low_dim_X_test, self.low_dim_Y_train, self.low_dim_Y_test = \
            creditdata.dimension_decrease()
        x1, x2, y1, y2 = self.low_dim_X_train, self.low_dim_X_test, self.low_dim_Y_train, self.low_dim_Y_test
        self.discretizer = PostprocessAlt(x1, x2, y1, y2)
        self.discretized_X_train, self.discretized_X_test, self.discretized_Y_train, self.discretized_Y_test = \
            self.discretizer.improve_data()
        self.buildclf()
        self.buildprocessor()
        self.logfile = open("execution_Log", "a")

    def buildclf(self):
        self.classifier.append(processor(GaussianNB(), "Gaussian NB"))
        self.classifier.append(processor(KNeighborsClassifier(n_neighbors=15), "K Nearest Neighbors"))
        self.classifier.append(processor(SVC(), "C-Support Vector"))
        self.classifier.append(processor(LogisticRegression(), "Logistic Regression"))
        # self.classifier.append(processor(LinearDiscriminantAnalysis(), "Discriminant Analysis"))
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
        # print("Mislabeled points (%s Classification) out of a total %d points : %d" % (clf.descr, totaltest,mislabeled))
        Precision = 1 - mislabeled / totaltest
        # print("Precision of %s is %4.2f%%" % (clf.descr, Precision * 100))
        return Precision

    def compare_clf_and_prep(self, x1_raw, x2_raw, y1_raw, y2_raw, x1, x2, y1, y2):

        result_matrix = []
        maxprec = [0, "", ""]
        for clf in self.classifier:
            row = []
            prec = self.getprecision(clf, x1_raw, x2_raw, y1_raw, y2_raw)
            if prec > maxprec[0]:
                maxprec = [prec, "no preprocess", clf.descr]
            row.append(prec)
            for processor in self.processor:
                processed_X_train = processor.obj.fit_transform(x1)
                processed_X_test = processor.obj.fit_transform(x2)
                prec = self.getprecision(clf, processed_X_train, processed_X_test, y1, y2)
                if prec > maxprec[0]:
                    maxprec = [prec, processor.descr, clf.descr]
                row.append(prec)
            result_matrix.append(row)
        self.printresult(result_matrix)
        print("The maximum precision is %0.8f%%, with %s and %s classification" %(maxprec[0]*100, maxprec[1],
                                                                                 maxprec[2]))
        self.logfile.write("%0.8f%%, %s and %s classification" %(maxprec[0]*100, maxprec[1], maxprec[2]))
        self.logfile.write("\n")


    def printresult(self, matrix):
        print("%28s%28s" %("___________________________", "No preprocess"), end='')
        for p in self.processor:
            print("%28s" %p.descr, end='')
        print('')
        for i in range(len(self.classifier)):
            print("%28s" % self.classifier[i].descr, end='')
            for j in range(len(self.processor)+1):
                print("%22s%4.2f%%" % ("", matrix[i][j]*100), end='')
            print('')

    def comparison(self, x1_raw, x2_raw, y1_raw, y2_raw, x1, x2, y1, y2, dx1, dx2, dy1, dy2):
        c = self.classifier[2]
        s = self.processor[5]

        preprocessed_X_train = s.obj.fit_transform(x1)
        preprocessed_X_test = s.obj.fit_transform(x2)
        preprocessed_dX_train = s.obj.fit_transform(dx1)
        preprocessed_dX_test = s.obj.fit_transform(dx2)

        comparison_result = [self.getprecision(c, x1_raw, x2_raw, y1_raw, y2_raw),
                             self.getprecision(c, preprocessed_X_train, preprocessed_X_test, y1, y2),
                             self.getprecision(c, preprocessed_dX_train, preprocessed_dX_test, dy1, dy2),
                             self.getprecision(c, dx1, dx2, dy1, dy2)]
        print("____________   Raw data        LD with S         LD with S&HL         Ld with HL")
        print("Precision        ", end="")
        for item in comparison_result:
            print("%4.2f%%%12s" % (item*100, ""), end="")
        print("")

    def run(self):
        self.compare_clf_and_prep(self.raw_X_train, self.raw_X_test, self.raw_Y_train, self.raw_Y_test,
                                  self.low_dim_X_train, self.low_dim_X_test, self.low_dim_Y_train,
                                  self.low_dim_Y_test)
        print("Comparison between raw data and preprocessed data in SVC with Non-linear transformation")
        self.comparison(self.raw_X_train, self.raw_X_test, self.raw_Y_train, self.raw_Y_test,
                        self.low_dim_X_train, self.low_dim_X_test, self.low_dim_Y_train,
                        self.low_dim_Y_test, self.discretized_X_train, self.discretized_X_test, self.discretized_Y_train,
                        self.discretized_Y_test)


class processor:
    def __init__(self, obj, str):
        self.obj = obj
        self.descr = str


if __name__ == '__main__':
    import time
    start_time = time.time()
    a = experiment()
    a.run()
    print("--- %d seconds ---" % (time.time() - start_time))



# print(__doc__)
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import QuantileTransformer


class ploting:
    def __init__(self, classifier, preprocessor, sample_points, x_axi_attr_index, y_axi_attr_index, title, xlabel,
                 ylabel):
        self.sample = sample_points
        self.prep = preprocessor
        self.x_attri1 = x_axi_attr_index
        self.x_attri2 = y_axi_attr_index
        self.X, self.y = self.load_data()
        self.model = classifier.fit(self.X, self.y)
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def make_meshgrid(self, x, y, h=.02):

        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(self, ax, clf, xx, yy, **params):
        """Plot the decision boundaries for a classifier.

        Parameters
        ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    def data_simplification(self, x_read, y_read):
        half_index = int(self.sample/2)
        simplified_x = np.empty([self.sample, 2], dtype=float)
        simplified_y = np.empty([self.sample], dtype=float)
        list_nondefault = range(0, int(len(y_read)/2))
        list_default = range(int(len(y_read)/2), int(len(y_read)))
        random_nondefault = random.sample(list_nondefault, half_index)
        random_default = random.sample(list_default, half_index)
        for i in range(0,half_index):
            simplified_x[i,0] = x_read[random_nondefault[i], self.x_attri1]
            simplified_x[i,1] = x_read[random_nondefault[i], self.x_attri2]
            simplified_y[i] = y_read[random_nondefault[i]]
            simplified_x[i + half_index, 0] = x_read[random_default[i], self.x_attri1]
            simplified_x[i + half_index, 1] = x_read[random_default[i], self.x_attri2]
            simplified_y[i + half_index] = y_read[[random_default[i]]]

        return simplified_x, simplified_y

    # # import some data to play with
    # iris = datasets.load_iris()
    # # Take the first two features. We could avoid this by using a two-dim dataset
    # X = iris.data[:, :2]
    # y = iris.target
    def load_data(self):
        from Preprocessing import Preprocess
        creditdata = Preprocess("default of credit card clients.xls")
        X_train, X_test, y_train, y_test = creditdata.load_dataset()
        return self.data_simplification(self.prep.fit_transform(X_train[:, :]), y_train)

    def plot(self):
        X0, X1 = self.X[:, 0], self.X[:, 1]
        xx, yy = self.make_meshgrid(X0, X1)

        ax = plt.gca()
        self.plot_contours(plt, self.model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=self.y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(self.title)

        plt.show()


if __name__ == '__main__':
    c = svm.SVC()
    p = QuantileTransformer()
    myplot = ploting(c, p, 20, 2, 4, "SVM Classifier", "x axis: edu", "y axis: age")
    myplot.plot()

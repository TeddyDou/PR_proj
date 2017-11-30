# print(__doc__)
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import QuantileTransformer


class ploting:
    def __init__(self, classifier, preprocessor):
        self.prep = preprocessor
        self.model = classifier


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

    def data_simplification(self, x_read, y_read, sample, x_axi_attr_index, y_axi_attr_index):
        half_index = int(sample/2)
        simplified_x = np.empty([sample, 2], dtype=float)
        simplified_y = np.empty([sample], dtype=float)
        list_nondefault = range(0, int(len(y_read)/2))
        list_default = range(int(len(y_read)/2), int(len(y_read)))
        random_nondefault = random.sample(list_nondefault, half_index)
        random_default = random.sample(list_default, half_index)
        for i in range(0,half_index):
            simplified_x[i,0] = x_read[random_nondefault[i], x_axi_attr_index]
            simplified_x[i,1] = x_read[random_nondefault[i], y_axi_attr_index]
            simplified_y[i] = y_read[random_nondefault[i]]
            simplified_x[i + half_index, 0] = x_read[random_default[i], x_axi_attr_index]
            simplified_x[i + half_index, 1] = x_read[random_default[i], y_axi_attr_index]
            simplified_y[i + half_index] = y_read[[random_default[i]]]

        return simplified_x, simplified_y

    def load_data(self, sample, x_axi_attr_index, y_axi_attr_index):
        from PreprocessingAlt import PreprocessAlt
        creditdata = PreprocessAlt("default of credit card clients.xls")
        raw_X_train, raw_X_test, raw_y_train, raw_y_test = creditdata.load()
        low_dim_X_train, low_dim_X_test, low_dim_Y_train, low_dim_Y_test = creditdata.dimension_decrease()
        return self.data_simplification(self.prep.fit_transform(low_dim_X_train), low_dim_Y_train, sample,
                                        x_axi_attr_index, y_axi_attr_index)

    def plot(self, sample_points, x_axi_attr_index, y_axi_attr_index, title, xlabel, ylabel):
        plot_x, plot_y = self.load_data(sample_points, x_axi_attr_index, y_axi_attr_index)
        X0, X1 = plot_x[:, 0], plot_x[:, 1]
        xx, yy = self.make_meshgrid(X0, X1)

        plt.figure()
        ax = plt.gca()
        self.plot_contours(plt, self.model.fit(plot_x, plot_y), xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=plot_y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    def show(self):
        plt.show()


if __name__ == '__main__':
    c = svm.SVC()
    p = QuantileTransformer()
    myplot = ploting(c, p)

    # column 0: limited balance, 1: sex(gender) 2: education 3: marriage 4: age 5: amount owed 6: missed payment
    myplot.plot(20, 0, 1, "SVM Classifier", "x axis: limited balance", "y axis: gender")
    myplot.plot(20, 2, 5, "SVM Classifier", "x axis: education", "y axis: amount owed")
    myplot.plot(20, 4, 6, "SVM Classifier", "x axis: age", "y axis: missed payment")
    myplot.show()
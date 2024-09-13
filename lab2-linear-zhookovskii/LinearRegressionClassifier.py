import numpy as np
from sklearn.base import BaseEstimator


class LinearRegressionClassifier(BaseEstimator):

    def __init__(self, tau=0):
        self.theta = None
        self.tau = tau
        self.from_classes = dict()
        self.to_classes = dict()

    def fit(self, X_train, y_train):
        classes = np.unique(y_train)
        if classes.shape[0] != 2:
            raise ValueError('this estimator should only be used for binary classification')
        self.from_classes = {
            classes[0]: -1,
            classes[1]: 1
        }
        self.to_classes = {
            -1: classes[0],
            1: classes[1]
        }

        n, m = np.array(X_train).shape
        F = np.c_[np.array(X_train), np.ones(n)]
        y = np.array(
            list(map(lambda x: self.from_classes[x], y_train))
        )
        Im = np.eye(m + 1)
        Im[m][m] = 0
        A = np.dot(F.T, F)
        B = self.tau * Im
        self.theta = np.dot(
            np.dot(
                np.linalg.inv(
                    A + B
                ),
                F.T
            ),
            y
        )

    def predict(self, X_test):
        y_test = []
        for x in X_test:
            vec = np.append(x, 1)
            y_test.append(
                self.to_classes[-1 if np.dot(self.theta, vec) < 0 else 1]
            )
        return np.array(y_test)

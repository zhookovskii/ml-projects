import numpy as np
from math import exp
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


class GDClassifier(BaseEstimator):

    valid_loss_function = {'mse', 'log', 'sigmoid'}
    valid_regularization = {'ridge', 'lasso', 'elastic'}

    def __init__(
            self,
            lr=0.001,
            epochs=100,
            eps=1e-3,
            loss_function='mse',
            regularization=None,
            alpha=0,
            beta=0,
            track_loss=False,
            track_score=False,
    ):
        self.theta = None
        self.lr = lr
        self.epochs = epochs
        self.eps = eps
        if loss_function not in self.valid_loss_function:
            raise ValueError(f'{loss_function} is not a valid loss function')
        self.loss_function = loss_function
        if regularization is None:
            self.alpha = 0
            self.beta = 0
        elif regularization not in self.valid_regularization:
            raise ValueError(f'{regularization} is not a valid regularization')
        elif regularization == 'ridge':
            self.alpha = 0
            self.beta = beta
        elif regularization == 'lasso':
            self.alpha = alpha
            self.beta = 0
        else:
            self.alpha = alpha
            self.beta = beta
        self.regularization = regularization
        self.from_classes = dict()
        self.to_classes = dict()

        self.track_loss = track_loss
        self.loss_values = []
        self.completed_epochs = 0

        self.track_score = track_score
        self.score_values = []

    def _loss(self, x_i, y_i):
        if self.loss_function == 'mse':
            return -2 * x_i * (y_i - np.dot(self.theta, x_i))
        elif self.loss_function == 'log':
            return -y_i * x_i / (exp(y_i * np.dot(self.theta, x_i)) + 1)
        else:
            m = y_i * np.dot(self.theta, x_i)
            return -y_i * x_i * exp(m) / (exp(2 * m) + 2 * exp(m) + 1)

    def _calculate_loss(self, X, y):
        n, m = X.shape
        l1 = self.alpha * sum(map(abs, self.theta))
        l2 = self.beta * sum(map(lambda x: x ** 2, self.theta))
        if self.loss_function == 'mse':
            return sum((y[i] - np.dot(self.theta, X[i])) ** 2 for i in range(n)) + l1 + l2
        elif self.loss_function == 'log':
            return sum(np.log(1 + np.exp(-y[i] * np.dot(self.theta, X[i]))) for i in range(n)) + l1 + l2
        else:
            return sum(1 / (1 + exp(y[i] * np.dot(self.theta, X[i]))) for i in range(n)) + l1 + l2

    def _calculate_score(self, X_test, y_test):
        pred = self.predict(X_test)
        return accuracy_score(y_test, pred)

    def fit(self, X_train, y_train, X_test=None, y_test=None):
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
        X = np.c_[np.array(X_train), np.ones(n)]
        y = np.array(
            list(map(lambda x: self.from_classes[x], y_train))
        )
        self.theta = np.zeros(m + 1)

        for _ in range(self.epochs):
            l1 = (self.alpha / 2) * np.array([-1 if a < 0 else 1 for a in self.theta])
            l2 = self.beta * self.theta
            step = sum(self._loss(X[i], y[i]) for i in range(n)) + l1 + l2

            if self.track_loss:
                self.loss_values.append(self._calculate_loss(X, y))
            if self.track_score:
                self.score_values.append(self._calculate_score(X_test, y_test))
            self.completed_epochs += 1

            new_theta = self.theta - self.lr * step
            self.theta = new_theta

    def predict(self, X_test):
        y_test = []
        for x in X_test:
            vec = np.append(x, 1)
            y_test.append(
                self.to_classes[-1 if np.dot(self.theta, vec) < 0 else 1]
            )
        return np.array(y_test)

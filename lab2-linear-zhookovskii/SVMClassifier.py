from math import exp
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score


class SVMClassifier(BaseEstimator):

    valid_kernel = {'sqr', 'sigmoid', 'radial'}

    @staticmethod
    def _product(x, y):
        return np.dot(x, y)

    @staticmethod
    def _sqr_product(x, y):
        return np.dot(x, y) ** 2

    @staticmethod
    def _sigmoid_product(x, y):
        return 2 / (1 + exp(np.dot(x, y)))

    def _radial_basis_product(self, x, y):
        return exp(-self.beta * np.dot(x - y, x - y))

    def __init__(
            self,
            C=0.5,
            eps=1e-3,
            max_passes=1,
            epochs=100,
            kernel=None,
            beta=0.0,
            track_loss=False,
            track_score=False
    ):
        self.C = C
        self.eps = eps
        self.max_passes = max_passes
        self.epochs = epochs
        self.beta = beta
        if kernel is not None and kernel not in self.valid_kernel:
            raise ValueError(f'{kernel} is not a valid kernel')
        self.kernel = kernel
        self.from_classes = dict()
        self.to_classes = dict()
        self.alpha = None
        self.b = None
        self.X = None
        self.y = None
        self.n = None

        self.track_loss = track_loss
        self.loss_values = []
        self.completed_epochs = 0

        self.track_score = track_score
        self.score_values = []

    @staticmethod
    def _calculate_loss(X, y, alpha, kernel):
        n, m = X.shape
        snd = sum(sum(alpha[i] * alpha[j] * y[i] * y[j] * kernel(X[i], X[j]) for j in range(n)) for i in range(n))
        return -sum(alpha[i] for i in range(n)) + 1 / 2 * snd

    def _calculate_score(self, X, y, alpha, b, kernel, X_test, y_test):
        n, m = X.shape
        pred = []
        for x in X_test:
            value = sum(alpha[i] * y[i] * kernel(X[i], x) for i in range(n)) + b
            pred.append(
                self.to_classes[-1 if value < 0 else 1]
            )
        return accuracy_score(pred, y_test)

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

        X = np.array(X_train)
        n, m = X.shape
        y = np.array(
            list(map(lambda x: self.from_classes[x], y_train))
        )
        alpha = np.zeros(n)
        b = 0
        passes = 0

        if self.kernel is None:
            kernel = self._product
        elif self.kernel == 'sqr':
            kernel = self._sqr_product
        elif self.kernel == 'sigmoid':
            kernel = self._sigmoid_product
        else:
            kernel = self._radial_basis_product

        def margin_error(idx):
            return sum(alpha[l] * y[l] * kernel(X[l], X[idx]) for l in range(n)) + b - y[idx]

        while passes < self.max_passes and self.completed_epochs < self.epochs:
            changed = 0
            for i in range(n):
                if self.completed_epochs >= self.epochs:
                    break
                E_i = margin_error(i)
                if (y[i] * E_i < -self.eps and alpha[i] < self.C) or (y[i] * E_i > self.eps and alpha[i] > 0):
                    while (j := np.random.randint(0, n)) == i:
                        pass
                    if y[i] == y[j]:
                        L, H = max(0, alpha[i] + alpha[j] - self.C), min(self.C, alpha[i] + alpha[j])
                    else:
                        L, H = max(0, alpha[j] - alpha[i]), min(self.C, self.C + alpha[j] - alpha[i])
                    if L == H:
                        continue
                    dot_ij = kernel(X[i], X[j])
                    dot_ii = kernel(X[i], X[i])
                    dot_jj = kernel(X[j], X[j])
                    eta = 2 * dot_ij - dot_ii - dot_jj
                    if eta == 0:
                        continue
                    E_j = margin_error(j)
                    new_alpha_j = alpha[j] - y[j] * (E_i - E_j) / eta
                    if new_alpha_j < L:
                        new_alpha_j = L
                    if new_alpha_j > H:
                        new_alpha_j = H
                    if abs(new_alpha_j - alpha[j]) < 1e-5:
                        continue
                    new_alpha_i = alpha[i] + y[i] * y[j] * (alpha[j] - new_alpha_j)
                    b_1 = b - E_i - y[i] * (new_alpha_i - alpha[i]) * dot_ii - y[j] * (new_alpha_j - alpha[j]) * dot_ij
                    b_2 = b - E_j - y[i] * (new_alpha_i - alpha[i]) * dot_ij - y[j] * (new_alpha_j - alpha[j]) * dot_jj
                    if 0 < new_alpha_i < self.C:
                        b = b_1
                    elif 0 < new_alpha_j < self.C:
                        b = b_2
                    else:
                        b = (b_1 + b_2) / 2
                    alpha[i] = new_alpha_i
                    alpha[j] = new_alpha_j
                    changed += 1

                    if self.track_loss:
                        self.loss_values.append(self._calculate_loss(X, y, alpha, kernel))
                    if self.track_score:
                        self.score_values.append(self._calculate_score(X, y, alpha, b, kernel, X_test, y_test))
                    self.completed_epochs += 1
            if changed == 0:
                passes += 1
            else:
                passes = 0

        self.alpha = alpha
        self.b = b
        self.X = X
        self.y = y
        self.n = n

    def fit_quadratic(self, X_train, y_train):
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

        X = np.array(X_train)
        n, m = X.shape
        y = np.array(
            list(map(lambda x: self.from_classes[x], y_train))
        )
        alpha = np.zeros(n)
        b = 0

        def condition():
            nonlocal b
            w = sum(alpha[i] * X[i] * y[i] for i in range(n))
            min_positive = min(self.kernel(w, X[i]) if y[i] == 1 else np.inf for i in range(n))
            max_negative = max(self.kernel(w, X[i]) if y[i] == -1 else -np.inf for i in range(n))
            b = -1 / 2 * (min_positive + max_negative)

            def check_alpha(idx):
                if alpha[idx] == 0:
                    return y[idx] * (self.kernel(w, X[idx]) + b) >= 1 - self.eps
                elif alpha[idx] == self.C:
                    return y[idx] * (self.kernel(w, X[idx]) + b) <= 1 + self.eps
                else:
                    return 1 - self.eps < y[idx] * (self.kernel(w, X[idx]) + b) < 1 + self.eps

            return all(check_alpha(i) for i in range(n))

        def objective(u, v, x=None):
            c = sum(alpha[i] for i in fixed)
            c += sum(sum(alpha[i] * alpha[j] * y[i] * y[j] * self.kernel(X[i], X[j]) for j in fixed) for i in fixed)
            c += zeta
            c += n / 2 * (zeta ** 2) * y[v] * self.kernel(X[v], X[v])

            b = y[u] - y[v]
            b += n * zeta * y[u] * y[v] * self.kernel(X[u], X[v])
            b += n / 2 * zeta * (y[u] ** 2) * self.kernel(X[u], X[u])
            b -= n * zeta * y[u] * y[v] * self.kernel(X[v], X[v])

            a = -n * (y[u] ** 2) * y[v] * self.kernel(X[u], X[v])
            a -= n / 2 * (y[u] ** 3) * self.kernel(X[u], X[u])
            a += n / 2 * (y[u] ** 2) * y[v] * self.kernel(X[v], X[v])

            if x is None:
                if a == 0:
                    return -np.inf, x
                x = -b / (2 * a)
            return a * (x ** 2) + b * x + c, x

        indices = list(range(n))
        iterations = 0

        while not condition() and iterations < 100:
            r, s = np.random.choice(indices, size=2, replace=False)
            fixed = list(filter(lambda i: i != r and i != s, range(n)))
            zeta = -sum(alpha[i] * y[i] for i in fixed)

            if y[r] == y[s]:
                left, right = max(0, -y[r] * (y[s] * self.C - zeta)), min(self.C, y[r] * zeta)
            else:
                left, right = max(0, y[r] * zeta), min(self.C, -y[r] * (y[s] * self.C - zeta))

            candidates = [objective(r, s, x=left), objective(r, s, x=right), objective(r, s)]
            _, new_r = max(candidates, key=lambda t: t[0])
            new_s = (zeta - new_r * y[r]) / y[s]
            alpha[r] = new_r
            alpha[s] = new_s
            iterations += 1
            print(iterations)

        self.alpha = alpha
        self.b = b
        self.X = X
        self.y = y
        self.n = n

    def predict(self, X_test):
        if self.kernel is None:
            kernel = self._product
        elif self.kernel == 'sqr':
            kernel = self._sqr_product
        elif self.kernel == 'sigmoid':
            kernel = self._sigmoid_product
        else:
            kernel = self._radial_basis_product
        y_test = []
        for x in X_test:
            value = sum(self.alpha[i] * self.y[i] * kernel(self.X[i], x) for i in range(self.n)) + self.b
            y_test.append(
                self.to_classes[-1 if value < 0 else 1]
            )
        return np.array(y_test)

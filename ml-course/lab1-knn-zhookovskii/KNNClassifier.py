from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
import numpy as np
from math import sqrt, exp, pi
from sklearn.metrics import DistanceMetric
from itertools import groupby


class KNNClassifier(BaseEstimator):

    valid_metrics = {"minkowski", "chebyshev", "cosine"}
    valid_kernels = {"uniform", "triangular", "epanechnikov", "gaussian"}

    def __init__(
            self,
            k: int = 5,
            window_size: float = -1.0,
            metric: str = "minkowski",
            p: int = 2,
            kernel: str = "uniform",
            leaf_size: int = 20,
            weights=None,
            obj_weights=None
    ):
        self.k = k
        if window_size < 0:
            self.fixed = False
        else:
            self.fixed = True
        self.window_size = window_size
        self.p = p
        self.leaf_size = leaf_size

        if metric not in self.valid_metrics:
            raise ValueError("unknown metric:", metric)
        if kernel not in self.valid_kernels:
            raise ValueError("unknown kernel:", kernel)
        if metric == "minkowski" and p <= 0:
            raise ValueError(
                "parameter p for minkowski distance must be a positive integer, use 'chebyshev' for p=+inf"
            )

        self.metric = metric
        self.kernel = kernel
        self.tree = None
        self.X_train = None
        self.y_train = None
        self.weights = weights
        self.obj_weights = obj_weights
        self.outlier_label = None

    def fit(self, X_train, y_train, outlier_label=None):
        self.outlier_label = outlier_label
        if self.weights is None:
            self.weights = [1] * X_train.shape[1]
        if self.obj_weights is None:
            self.obj_weights = [1] * X_train.shape[0]

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        if self.metric == "cosine":
            distance_metric = DistanceMetric.get_metric('pyfunc', func=self._cosine)
        elif self.metric == "chebyshev":
            distance_metric = DistanceMetric.get_metric('pyfunc', func=self._chebyshev)
        else:
            distance_metric = DistanceMetric.get_metric('minkowski', p=self.p, w=self.weights)

        self.tree = BallTree(X_train, leaf_size=self.leaf_size, metric=distance_metric)

    def _get_nearest(self, x):
        if self.fixed:
            indices, dist = self.tree.query_radius([x], r=self.window_size, return_distance=True)
            boundary = self.window_size
            indices = indices[0]
            dist = dist[0]
        else:
            dist, indices = self.tree.query([x], k=self.k + 1)
            boundary = dist[0][-1]
            indices = indices[0][:-1]
            dist = dist[0][:-1]
        if boundary == 0:
            boundary = 1
        return list(zip(indices, dist / boundary))

    def _get_kernel_sum(self, dist):
        if self.kernel == 'uniform':
            kernel_function = self._uniform_kernel
        elif self.kernel == 'triangular':
            kernel_function = self._triangular_kernel
        elif self.kernel == 'epanechnikov':
            kernel_function = self._epanechnikov_kernel
        else:
            kernel_function = self._gaussian_kernel
        return sum(map(lambda t: kernel_function(t[1]) * self.obj_weights[t[0]], dist))

    def predict(self, X_test):
        y_test = []
        for x in X_test:
            nearest = self._get_nearest(x)
            nearest = sorted(nearest, key=lambda t: self.y_train[t[0]])
            grouped = {
                k: list(v) for k, v in groupby(nearest, key=lambda t: self.y_train[t[0]])
            }
            summed = list(map(lambda group: (group[0], self._get_kernel_sum(group[1])), grouped.items()))

            if len(summed) == 0:
                if self.outlier_label is None:
                    raise ValueError('cannot assign label to object with no neighbors')
                else:
                    label = self.outlier_label
            else:
                label, _ = max(summed, key=lambda entry: entry[1])

            y_test.append(label)
        return np.array(y_test)

    def _cosine(self, a, b):
        n = len(a)
        num = sum(a[i] * b[i] * self.weights[i] for i in range(n))
        den = sqrt(
            sum((a[i] ** 2) * self.weights[i] for i in range(n)) * sum((b[i] ** 2) * self.weights[i] for i in range(n))
        )
        return 1 - num / den

    def _chebyshev(self, a, b):
        return max(abs(a[i] - b[i]) * self.weights[i] for i in range(len(a)))

    @staticmethod
    def _uniform_kernel(distance):
        return 1 / 2 * (distance < 1)

    @staticmethod
    def _triangular_kernel(distance):
        return (1 - abs(distance)) * (distance < 1)

    @staticmethod
    def _epanechnikov_kernel(distance):
        return 3 / 4 * (1 - distance ** 2) * (distance < 1)

    @staticmethod
    def _gaussian_kernel(distance):
        return 1 / (sqrt(2 * pi)) * exp(-(distance ** 2) / 2)

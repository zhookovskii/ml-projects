import numpy as np
from BaseSelector import BaseSelector
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


class WrapperSelector(BaseSelector):

    def __init__(self, model, min_features: int = 2, n_shuffles: int = 5):
        super().__init__(model, min_features)
        self.n_shuffles = n_shuffles

    def fit_transform(self, X, y):
        n, m = X.shape
        X = X.toarray()
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7)

        self.feature_importance = np.zeros(m)
        self.model.fit(X_train, y_train)
        baseline = self.model.score(X_valid, y_valid)

        for i in range(m):
            performance_delta = 0
            for _ in range(self.n_shuffles):
                col = X_valid[:, i].copy()
                np.random.shuffle(X_valid[:, i])
                score = self.model.score(X_valid, y_valid)
                X_valid[:, i] = col
                performance_delta = max(max(baseline - score, 0), performance_delta)
            if i % 100 == 0:
                print(f"estimated first {i} features")
            self.feature_importance[i] = performance_delta

        sorted_features = np.argsort(self.feature_importance)
        self.leave_features = np.array(list(filter(
            lambda x: self.feature_importance[x] != 0,
            sorted_features[m - self.min_features:]
        )))
        return csr_matrix(X[:, self.leave_features])

import numpy as np
from sklearn.linear_model.ridge import Ridge


class Arima:

    """ Learn an average of rank number of previous points."""

    @staticmethod
    def find_closest_vals(X, x, y, r):
        """ Values of r closest points to x in X, s.t. x < X."""
        ret = np.zeros((r,))
        diffs = (X - x).ravel().astype(float)
        diffs[diffs >= 0] = np.inf
        diffs = np.abs(diffs)
        count = 0
        while count < r:
            v, i = np.min(diffs), np.argmin(diffs)
            ret[count] = y[i] if np.isfinite(v) else 0
            diffs[i] = np.inf
            count += 1
        return ret

    def __init__(self, rank, **kwargs):
        self.rank = rank
        self.kwargs = kwargs
        self.ridge = None
        self.X = None
        self.y = None
        self.bias = None

    def fit(self, X, y):
        """
        X is an arbitrary 1D signal.
        For each x in X, take up to 7 previous values and fit using Ridge regression.
        Assume the mean is 0.
        """
        self.bias = y.mean()
        y = y - self.bias

        self.X = X
        self.y = y
        M = np.zeros((X.shape[0], self.rank))
        for xi, x in enumerate(X):
            M[xi, :] = Arima.find_closest_vals(X, x, y, r=self.rank)
        self.ridge = Ridge(**self.kwargs)
        self.ridge.fit(M, y)

    def predict(self, X):
        """ Predict from closest vals. """
        M = np.zeros((X.shape[0], self.rank))
        for xi, x in enumerate(X):
            M[xi, :] = Arima.find_closest_vals(self.X, x, self.y, r=self.rank)
        return self.bias + self.ridge.predict(M)

def test_arima():
    import matplotlib.pyplot as plt
    n = 100
    noise = 0.7
    X = np.linspace(-10, 10, n).reshape((n, 1))
    y = np.sin(X) + noise * np.random.rand(n, 1)
    model = Arima(rank=7)
    model.fit(X, y)
    yp = model.predict(X)
    plt.figure()
    plt.plot(X, y, ".")
    plt.plot(X, yp, "-")


if __name__ == "__main__":
    test_arima()
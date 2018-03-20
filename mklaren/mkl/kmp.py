import numpy as np
from numpy.linalg import norm
from warnings import warn
from sklearn.linear_model.ridge import Ridge
from ..kernel.kinterface import Kinterface


class KMP:
    """
        General matching pursuit algorithm with lookahead, extended depending on the
        cost function. Sampling of the basis funtions from multiple kernels.
    """

    def __init__(self, rank, delta, lbd=1, tol=1e-10):
        self.rank = rank
        self.delta = delta
        self.Acts = []
        self.Ks = []
        self.sol_path = []
        self.tol = tol
        self.lbd = lbd
        self.ridge = None
        self.trained = False

    @staticmethod
    def gain(X, r):
        """ Gain of the columns in X with respect to residual r. """
        xn = norm(X, axis=0)
        rn = norm(r)
        return norm(X.T.dot(r)) / (xn * rn)

    def fit(self, Ks, y):
        assert all([isinstance(K, Kinterface) for K in Ks])

        n = Ks[0].shape[0]
        if self.rank > n:
            msg = "Rank is larger than n, %d > %d" % (self.rank, n)
            self.rank = n
            warn(msg)

        k = len(Ks)
        Acts = [list() for K in Ks]
        Inas = [range(n) for K in Ks]

        r = y.reshape((n, 1)) - np.mean(y)
        mu = np.zeros((n, 1))
        sol_path = np.zeros((self.rank, n))

        gains = np.zeros((k, n))
        for step in range(self.rank):
            gains[:, :] = 0
            for ki, K in enumerate(Ks):
                if len(Inas[ki]) == 0:
                    continue
                inxs = np.random.choice(Inas[ki],
                                        size=min(self.delta, len(Inas[ki])),
                                        replace=False)
                gains[ki, inxs] = self.gain(K[:, inxs], r)
            if np.max(gains) < self.tol:
                msg = "Iterations ended prematurely at step = %d < %d" % (k, self.rank)
                self.rank = step
                warn(msg)
                break

            kern, pivot = np.unravel_index(np.argmax(gains), gains.shape)
            col = Ks[kern][:, pivot].reshape((n, 1))
            Acts[kern].append(pivot)
            Inas[kern].remove(pivot)

            update = col * (col.T.dot(r)) / (norm(col) ** 2)
            r = r - update
            mu = mu + update

            assert norm(col.T.dot(r)) < 1e-5
            sol_path[step] = mu.ravel()

        # Calculate
        self.ridge = Ridge(alpha=self.lbd, fit_intercept=True)
        self.ridge.fit(np.hstack([K[:, act] for K, act in zip(Ks, Acts) if len(act)]), y)
        self.Acts = Acts
        self.Ks = Ks
        self.sol_path = sol_path
        self.trained = True

    def predict(self, Xs):
        """ Predict values in Xs for stored kernels"""
        assert self.trained
        G = np.hstack([K(X, K.data[act, :]) for X, K, act in zip(Xs, self.Ks, self.Acts)
                       if len(act)])
        return self.ridge.predict(G)

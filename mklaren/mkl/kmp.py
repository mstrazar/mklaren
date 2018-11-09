import numpy as np
from numpy.linalg import norm
from warnings import warn
from ..kernel.kinterface import Kinterface


# TODO: compute L2-regularized LS path at each step.
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
        self.P = None
        self.sol_path = []
        self.tol = tol
        self.lbd = lbd
        self.ridge = None
        self.bias = None
        self.coef_path = None

    @staticmethod
    def gain(X, r):
        """ Gain of the columns in X with respect to residual r. """
        xn = norm(X, axis=0)
        rn = norm(r)
        return norm(X.T.dot(r)) / (xn * rn)

    def fit(self, Ks, y):
        assert all([isinstance(K, Kinterface) for K in Ks])
        self.bias = y.mean()
        y = y - self.bias

        n = Ks[0].shape[0]
        if self.rank > n:
            msg = "Rank is larger than n, %d > %d" % (self.rank, n)
            self.rank = n
            warn(msg)

        k = len(Ks)
        Acts = [list() for K in Ks]
        Inas = [range(n) for K in Ks]
        P = []

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
            P.append(kern)

            alpha = (col.T.dot(r)) / (norm(col) ** 2)
            update = col * alpha
            r = r - update
            mu = mu + update

            assert norm(col.T.dot(r)) < 1e-5
            sol_path[step] = mu.ravel()

        # Calculate least-squares fit
        self.Acts = Acts
        self.Ks = Ks
        self.P = np.array(P, dtype=int)
        self.sol_path = sol_path
        self._fit_path(y)

    def transform(self, Xs):
        """ Project test data into the subsampled space.
            Permute the columns w.r.t the selection order. """
        d2 = np.atleast_2d
        A = np.zeros((Xs[0].shape[0], self.rank))
        current = dict()
        for pi, p in enumerate(self.P):
            a = self.Acts[p][current.get(p, 0)]
            A[:, pi] = self.Ks[p](d2(Xs[p]), d2(self.Ks[p].data[a])).ravel()
            current[p] = current.get(p, 0) + 1
        return A

    def _fit_path(self, y):
        """ Compute least squares path for each column in the sequence.
            Note that columns are not decorellated in the selection (fit step). """
        coef_path = np.zeros((self.rank, self.rank))
        A = self.transform([K.data for K in self.Ks])
        for pi, p in enumerate(self.P):
            coef_path[pi, :pi+1] = np.linalg.lstsq(A[:, :pi+1], y, rcond=None)[0].ravel()
        self.coef_path = coef_path

    def predict(self, Xs):
        """ Predict values in Xs for stored kernels"""
        A = self.transform(Xs)
        return self.bias + A.dot(self.coef_path[-1])

    def predict_path(self, Xs):
        """ Predict whole regularization path."""
        A = self.transform(Xs)
        return self.bias + A.dot(self.coef_path.T)

from ..kernel.kinterface import Kinterface
import numpy as np


class ICD:

    """
    Incomplete Cholesky Decomposition.

    """

    def __init__(self, rank, eps=1e-10):
        """
        :param rank:
            Maximum allowed rank.
        :param eps:
            Allowed tolerance.
        """
        self.rank = rank
        self.eps = eps
        self.G = None
        self.trained = False

    def fit(self, K):
        """
        :param K:
            Kernel matrix.
        :return:
            Set G and active set.
        """
        n = K.shape[0]
        G = np.zeros((n, n))
        if isinstance(K, Kinterface):
            D = K.diag().copy()
        else:
            D = np.diag(K).copy()
        J = set(range(n))
        I = list()
        for k in range(n):
            # select pivot d
            i = np.argmax(D)
            I.append(i)
            J.remove(i)
            j = list(J)
            G[i, k] = np.sqrt(D[i])
            G[j, k] = 1.0 / G[i, k] * (K[j, i] - G[j, :].dot(G[i, :].T))
            D[j] = D[j] - (G[j, k]**2).ravel()

            # eliminate selected pivot
            D[i] = 0

            # check residual lower bound and maximum rank
            if np.max(D) < self.eps or k + 1 == self.rank:
                break

        self.active_set_ = I
        self.G = G[:, :k+1]
        self.trained = True


    def __call__(self, i, j):
        assert self.trained
        return self.G[i, :].dot(self.G[j, :].T)


    def __getitem__(self, item):
        assert self.trained
        return self.G[item[0], :].dot(self.G[item[1], :].T)


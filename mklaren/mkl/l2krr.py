

"""
Kernel alignment based on kernel Ridge regression. The kernels are not assumed to be centered.

    1. C. Cortes, L-2 Regularization for Learning Kernels. UAI (2009).

"""
from align import Align
from numpy import ones, zeros, eye, array, hstack, sqrt
from numpy.linalg import inv, norm
from ..util.la import woodbury_inverse

class L2KRR(Align):

    """
    :ivar Kappa: (``numpy.ndarray``) Combined kernel matrix.

    :ivar mu: (``numpy.ndarray``) Kernel weights.

    """

    def __init__(self, lbd=0.01, lbd2=1, eps = 0.01, max_iter=1000, nu=0.5):
        """
        :param lbd: (``float``) Regularization parameter (instance weights).

        :param lbd2: (``float``) Regularization parameter (> 0, kernel weights)

        :param eps: (``float``) Tolerance parameter (> 0).

        :param nu: (``float``) Interpolation parameter in (0, 1).

        :param max_iter: Maximum number of iterations.

        """
        self.lbd = lbd
        self.lbd2 = lbd2
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.mu = self.Kappa = self.trained = None
        self.alpha = self.iter = 0

    def fit(self, Ks, y, holdout=None, mu0=None):
        """
        Learn weights for kernel matrices or Kinterfaces.

        :param Ks: (``list``) of (``numpy.ndarray``) or of (``Kinterface``) to be aligned.

        :param y: (``numpy.ndarray``) Class labels :math:`y_i \in {-1, 1}` or regression targets.

        :param holdout: (``list``) List of indices to exlude from alignment.
        """
        m = len(y)
        p = len(Ks)
        y = y.reshape((m, 1))

        # Generalization to Kinterfaces
        Ks = [K[:, :] for K in Ks]

        if mu0 is None:
            mu0 = ones((p, 1))

        # Filter out hold out values
        if holdout is not None:
            holdin = sorted(list(set(range(m)) - set(holdout)))
            y = y[holdin]
            Ksa = list(map(lambda k: k[holdin, :][:, holdin], Ks))
            en = enumerate(Ksa)
            m = Ksa[0].shape[0]
        else:
            Ksa = Ks
            en = enumerate(Ksa)

        v = zeros((p, 1))
        mu = ones((p, 1))
        Kappa = array(sum([mu_i * k_i for mu_i, k_i in zip(mu0, Ksa)]))
        alpha1 = inv(Kappa + self.lbd * eye(m, m)).dot(y)

        for itr in range(self.max_iter):
            alpha = alpha1.copy()
            for i, K in en:
                 v[i] = alpha.T.dot(K).dot(alpha)
            mu = mu0 + self.lbd2 * (v / norm(v))

            Kappa = array(sum([mu_i * k_i for mu_i, k_i in zip(mu, Ksa)]))
            alpha1 = self.nu * alpha + \
                     (1 - self.nu) * inv(Kappa + self.lbd * eye(m, m)).dot(y)
            if norm(alpha1 - alpha) < self.eps:
                break

        self.mu = mu
        self.trained = True
        self.Kappa = array(sum([mu_i * k_i for mu_i, k_i in zip(mu, Ks)]))
        self.alpha = alpha
        self.iter = itr


class L2KRRlowRank(L2KRR):

    def fit(self, Vs, y, holdout=None, mu0=None):
        """
        Learn weights for kernel matrices or Kinterfaces.

        :param Vs: (``list``) of (``numpy.ndarray``) or of (``Kinterface``) to be aligned.

        :param y: (``numpy.ndarray``) Class labels :math:`y_i \in {-1, 1}` or regression targets.

        :param holdout: (``list``) List of indices to exlude from alignment.
        """
        assert self.lbd > 0
        m = len(y)
        p = len(Vs)
        y = y.reshape((m, 1))

        if mu0 is None:
            mu0 = ones((p, 1))

        # Filter out hold out values
        if holdout is not None:
            holdin = sorted(list(set(range(m)) - set(holdout)))
            y = y[holdin]
            Vs = list(map(lambda v: v[holdin, :], Vs))
            en = enumerate(Vs)
        else:
            Vs = Vs
            en = enumerate(Vs)

        V = hstack(Vs)
        n = V.shape[0]
        v = zeros((p, 1))
        mu = ones((p, 1))
        Kinv = woodbury_inverse(G=V * sqrt(mu0.T), sigma2=self.lbd)
        alpha1 = Kinv.dot(y).reshape(n, 1)

        for itr in range(self.max_iter):
            alpha = alpha1.copy()
            for i, u in en:
                v[i] = alpha.T.dot(u) ** 2
            mu = mu0 + self.lbd2 * (v / norm(v))
            Kinv = woodbury_inverse(G=V * sqrt(mu.T), sigma2=self.lbd)
            alpha1 = self.nu * alpha + (1 - self.nu) * Kinv.dot(y)
            if norm(alpha1 - alpha) < self.eps:
                break

        self.mu = mu
        self.trained = True
        self.alpha = alpha1
        self.iter = itr
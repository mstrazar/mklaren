

"""
Kernel alignment based on kernel Ridge regression. The kernels are not assumed to be centered.

    1. C. Cortes, L-2 Regularization for Learning Kernels. UAI (2009).

"""
from align import Align
from numpy import ones, zeros, eye, array
from numpy.linalg import inv, norm

class L2KRR(Align):

    """
    :ivar Kappa: (``numpy.ndarray``) Combined kernel matrix.

    :ivar mu: (``numpy.ndarray``) Kernel weights.

    """

    def __init__(self, lbd=0, lbd2=1, eps = 0.01, max_iter=1000, nu=0.5):
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
        self.alpha = self.itr = 0

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
        if not isinstance(holdout, type(None)):
            holdin = sorted(list(set(range(m)) - set(holdout)))
            y = y[holdin]
            Ksa = map(lambda k: k[holdin, :][:, holdin], Ks)
            en = enumerate(Ksa)
        else:
            Ksa = Ks
            en = enumerate(Ksa)

        v = zeros((p, 1))
        mu = ones((p, 1))
        Kappa = array(sum([mu_i * k_i for mu_i, k_i in zip(mu0, Ks)]))
        alpha1 = inv(Kappa + self.lbd * eye(m, m)).dot(y)

        for itr in range(self.max_iter):
            alpha = alpha1.copy()
            for i, K in en:
                 v[i] = alpha.T.dot(K).dot(alpha)
            mu = mu0 + self.lbd2 * (v / norm(v))

            Kappa = array(sum([mu_i * k_i for mu_i, k_i in zip(mu, Ks)]))
            alpha1 = self.nu * alpha + \
                     (1 - self.nu) * inv(Kappa + self.lbd * eye(m, m)).dot(y)
            if norm(alpha1 - alpha) < self.eps:
                break

        self.mu = mu
        self.trained = True
        self.Kappa = Kappa
        self.alpha = alpha
        self.iter = itr

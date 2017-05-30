"""

The Nystrom method learns a low-rank approximation of the kernel by evaluating the kernel function at a subset of data points.

    C. Williams and M. Seeger, "Using the Nystrom method to speed up kernel machines," in Proceedings of the 14th Annual Conference on Neural Information Processing Systems, 2001, no. EPFL-CONF.161322, pp. 682-688.

    A. Alaoui and M. Mahoney, "Fast Randomized Kernel Methods With Statistical Guarantees", arXiv, 2001.

Given a kernel matrix :math:`\mathbf{K} \in \mathbb{R}^{n\ x\ n}` and an active set :math:`\mathcal{A}`, the kernel matrix is approximated as

.. math::
    \mathbf{\hat{K}} = \mathbf{K}(:, \mathcal{A}) \mathbf{K}^{-1}(\mathcal{A}, \mathcal{A}) \mathbf{K}(:, \mathcal{A})^T

or in terms of :math:`\mathbf{G}`:

.. math::
    \mathbf{G} = \mathbf{K}(:, \mathcal{A}) \mathbf{K}^{-1/2}(\mathcal{A}, \mathcal{A})
"""


from ..kernel.kinterface import Kinterface
from sklearn.kernel_approximation import Nystroem
from numpy import array, diag, eye, real
from numpy.linalg import inv, cholesky
from numpy.random import choice, seed
from scipy.linalg import sqrtm

class Nystrom:

    """
    :ivar K: (``Kinterface``) or (``numpy.ndarray``) the kernel matrix.

    :ivar active_set_: The selected avtive set of indices.

    :ivar K_SS_i: (``numpy.ndarray``) the inverse kernel of the active set.

    :ivar K_XS: (``numpy.ndarray``) the kernel of the active set and the full data set.

    :ivar G: (``numpy.ndarray``) Low-rank approximation.
    """


    def __init__(self, rank=10, random_state=None, lbd=0, verbose=False):
        """
        :param rank: (``int``) Maximal decomposition rank.

        :param lbd: (``float``) regularization parameter (to be used in Kernel Ridge Regression).

        :param verbose (``bool``) Set verbosity.
        """
        self.trained = False
        self.rank = rank
        self.lbd = lbd
        self.verbose = verbose
        if random_state is not None:
            seed(random_state)


    def leverage_scores(self, K):
        """
        Compute leverage scores for matrix K and regularization parameter lbd.

        :param K: (``numpy.ndarray``) or of (``Kinterface``). The kernel to be approximated with G.

        :return: (``numpy.ndarray``) a vector of leverage scores to determine a sampling distribution.
        """
        dg = K.diag() if isinstance(K, Kinterface) else diag(K)
        pi = dg / dg.sum()
        n = K.shape[0]
        linxs = choice(xrange(n), size=self.rank, replace=True, p=pi)
        C = K[:, linxs]
        W = C[linxs, :]
        B = C.dot(real(sqrtm(W)))
        BTB = B.T.dot(B)
        BTBi = inv(BTB + n * self.lbd * eye(self.rank, self.rank))
        l = array([B[i, :].dot(BTBi).dot(B[i, :]) for i in xrange(n)])
        return l / l.sum()


    def fit(self, K, inxs=None):
        """
        Fit approximation to the kernel function / matrix.

        :param K: (``numpy.ndarray``) or of (``Kinterface``). The kernel to be approximated with G.

        :param inxs: (``list``) A predefined active set. If None, it is selected randomly, else use specified inputs and override the ``rank`` setting.
        """
        self.n       = K.shape[0]
        if inxs is None:
            if self.lbd == 0:
                if self.verbose: print("Choosing the active points randomly")
                inxs = choice(xrange(self.n),
                              size=self.rank, replace=False)

            else:
                if self.verbose: print("Choosing the active points via leverage scores")
                leverage = self.leverage_scores(K)
                inxs = choice(xrange(len(leverage)), size=self.rank, replace=False, p=leverage)

        self.rank    = len(inxs)
        self.K       = K
        self.active_set_  = inxs
        self.K_XS = K[:, inxs]
        self.K_SS = K[inxs, inxs]

        if len(inxs) > 1:
            self.K_SS_i = inv(K[inxs, :][:, inxs])
            R = sqrtm(self.K_SS_i)
            self.G = self.K_XS.dot(R)
        else:
            self.K_SS_i = array([1.0 / K[inxs[0], inxs[0]]])
            R = self.K_SS_i**0.5
            R = R.reshape((1, 1))
            self.G = self.K_XS.reshape((K.shape[0], 1)).dot(R)
            self.G = self.G.reshape((K.shape[0], 1))
        self.trained = True


    def predict(self, X=None, inxs=None):
        """ Predict values of the kernel for a test set.

        :param X:  (``numpy.ndarray``)  Test samples in the input space.

        :param inxs: (``list``) The active set.

        :return: (``numpy.ndarray``)  Predicted values of the kernel function against all training inputs.

        """
        if X is None and inxs is None:
            # Full approximation
            return self.K_XS.dot(self.K_SS_i).dot(self.K_XS.T)
        elif X is not None:
            # Invoke kernel directly
            K_YS = self.K.kernel(X, self.K.data[self.active_set_, :])
        elif inxs is not None:
            # Read existing values
            K_YS = self.K[inxs, self.active_set_]
        K_YS = K_YS.dot(self.K_SS_i).dot(self.K_XS.T)
        return K_YS


class NystromScikit:

    """
        Nystrom implementation form Scikit Learn wrapper.
        The main difference is in selection of inducing inputs.
    """

    def __init__(self, rank=10, random_state=42):
        """
        :param rank: (``int``) Maximal decomposition rank.

        :param random_state: (``int``) Random generator seed.
        """
        self.trained = False
        self.rank = rank
        self.random_state = random_state


    def fit(self, K, y):
        """
        Fit approximation to the kernel function / matrix.

        :param K: (``numpy.ndarray``) or of (``Kinterface``). The kernel to be approximated with G.

        :param y: (``numpy.ndarray``) Class labels :math:`y_i \in {-1, 1}` or regression targets.
        """
        assert isinstance(K, Kinterface)

        self.n           = K.shape[0]
        kernel           = lambda x, y: K.kernel(x, y, **K.kernel_args)
        self.model       = Nystroem(kernel=kernel,
                                    n_components=self.rank,
                                    random_state=self.random_state)

        self.model.fit(K.data, y)
        self.active_set_ = list(self.model.component_indices_[:self.rank])
        assert len(set(self.active_set_)) == len(self.active_set_) == self.rank
        R = self.model.normalization_
        self.G = K[:, self.active_set_].dot(R)
        self.trained = True








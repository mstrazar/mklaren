"""
A Ridge regression in the feature space induced by multiple kernel low-rank approximations.

Implemented for Nystrom and Cholesky-type decompositions.
"""


from ..mkl.align import Align, AlignLowRank
from ..mkl.alignf import Alignf, AlignfLowRank
from ..mkl.uniform import UniformAlignment, UniformAlignmentLowRank

from ..kernel.kernel import center_kernel_low_rank
from ..kernel.kinterface import Kinterface
from ..projection.icd import ICD
from ..projection.csi import CSI
from ..projection.nystrom import NystromScikit, Nystrom

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from numpy import array, hstack, sqrt, where, isnan
from numpy import hstack, array
from numpy.linalg import inv, norm


class RidgeMKL:
    """A MKL model in a transductive setting (test points are presented at training time).

    """

    mkls = {
        "align": Align,
        "alignf": Alignf,
        "alignfc": Alignf,
        "uniform": UniformAlignment,
    }

    mkls_low_rank = {
        "align": AlignLowRank,
        "alignf": AlignfLowRank,
        "alignfc": AlignfLowRank,
        "uniform": UniformAlignmentLowRank,
    }

    #  alignf expects kernels to be centered
    centered   = {"alignf", "alignfc"}
    supervised = {"align", "alignf", "alignfc"}

    def __init__(self, lbd=0, method="align", method_init_args={}, low_rank=False):
        """
        :param method: (``string``) "align", "alignf", or "uniform", MKL method to be used.

        :param low_rank: (``bool``) Use low-rank approximations.

        :param method_init_args: (``dict``) Initialization arguments for the MKL methods.

        :param lbd: (``float``) L2-regularization.
        """

        self.method  = method
        if not low_rank:
            self.mkl_model  = self.mkls[method](**method_init_args)
            if method == "alignfc":
                init_args = method_init_args.copy()
                init_args["typ"] = "convex"
                self.mkl_model  = self.mkls[method](**init_args)
        else:
            self.mkl_model  = self.mkls_low_rank[method](**method_init_args)
            if method == "alignfc":
                init_args = method_init_args.copy()
                init_args["typ"] = "convex"
                self.mkl_model  = self.mkls_low_rank[method](**init_args)
        self.lbd        = lbd
        self.low_rank   = low_rank
        self.trained    = False


    def fit(self, Ks, y, holdout=None):
        """Learn weights for kernel matrices or Kinterfaces.

        :param Ks: (``list``) of (``numpy.ndarray``) or of (``Kinterface``) to be aligned.

        :param y: (``numpy.ndarray``) Class labels :math:`y_i \in {-1, 1}` or regression targets.

        :param holdout: (``list``) List of indices to exlude from alignment.
        """

        # Expand kernel interfaces to kernel matrices
        expand = lambda K: K[:, :] if isinstance(K, Kinterface) else K
        Hs     = map(expand, Ks)

        # Assert correct dimensions
        assert Ks[0].shape[0] == len(y)

        # Fit MKL model
        if self.method in self.supervised:
            self.mkl_model.fit(Hs, y, holdout=holdout)
        else:
            self.mkl_model.fit(Hs)

        if self.low_rank:
            self.X = hstack(map(lambda e: sqrt(e[0]) * e[1],
                                zip(self.mkl_model.mu, Hs)))

            if self.method in self.centered:
                self.X = center_kernel_low_rank(self.X)
                self.X[where(isnan(self.X))] = 0

            # Fit ridge model with given lbd and MKL model
            self.ridge = KernelRidge(alpha=self.lbd,
                                     kernel="linear", )

            # Fit ridge on the examples minus the holdout set
            inxs = list(set(range(Hs[0].shape[0])) - set(holdout))
            self.ridge.fit(self.X[inxs], y[inxs])
            self.trained = True

        else:
            # Fit ridge model with given lbd and MKL model
            self.ridge = KernelRidge(alpha=self.lbd,
                                     kernel=self.mkl_model, )

            # Fit ridge on the examples minus the holdout set
            inxs = array(list(set(range(Hs[0].shape[0])) - set(holdout)))
            inxs = inxs.reshape((len(inxs), 1)).astype(int)
            self.ridge.fit(inxs, y[inxs])
            self.trained = True


    def predict(self, inxs):
        """
        Predict values for data on indices inxs (transcductive setting).

        :param inxs: (``list``) Indices of samples to be used for prediction.

        :return: (``numpy.ndarray``) Vector of prediction of regression targets.
        """
        assert self.trained

        if self.low_rank:
            return self.ridge.predict(self.X[inxs])
        else:
            inxs = array(inxs)
            inxs = inxs.reshape((len(inxs), 1)).astype(int)
            return self.ridge.predict(inxs).ravel()


class RidgeLowRank:

    """
    :ivar reg_model: (``sklearn.linear_model.Ridge``) regression model from Scikit.
    """

    # Static list of methods and their types
    CHOLESKY = "chol"
    NYSTROM  = "nystrom"
    methods  = {
        CHOLESKY: {"icd": ICD,
                   "csi": CSI,},
        NYSTROM: {"nystrom": NystromScikit, }
    }

    supervised = ["csi", "nystrom"]


    def __init__(self, method_init_args=None,
                 method="icd", lbd=0, rank=10, normalize=False):
        """
        Initialize object.

        :param method: (``string``) "icd", "csi", or "nystrom", low-rank method to be used.

        :param rank: (``int``) Maximal decomposition rank.

        :param normalize: (``bool``) Normalize data in Ridge model.

        :param method_init_args: (``dict``) Initialization arguments for the low-rank approximation models.

        :param lbd: (``float``) L2-regularization.
        """
        self.method     = method
        self.trained    = False

        # Determine method type
        if method_init_args is not None:
            self.method_init_args = method_init_args
        else:
            self.method_init_args = {}
        if self.method in self.methods[self.CHOLESKY]:
            self.type = self.CHOLESKY
            self.method_class = self.methods[self.CHOLESKY][method]
        elif self.method in self.methods[self.NYSTROM]:
            self.type = self.NYSTROM
            self.method_class = self.methods[self.NYSTROM][method]
        else:
            raise NotImplementedError

        # Initialize low-rank and regression model
        self.rank       = rank
        self.lr_models  = dict()
        self.reg_model  = Ridge(alpha=lbd, normalize=normalize)


    def fit(self, Ks, y, *method_args):
        """
        Fit multiple kernel functions.

        :param Ks: (``list``) of (``Kinterface``): List of kernel interfaces.

        :param y: (``numpy.ndarray``) Class labels :math:`y_i \in {-1, 1}` or regression targets.

        :param method_args: (``dict``) Arguments to the fit method of the kernel approximation method.
        """

        # Store kernels
        self.Ks = Ks

        # Set y as another argument
        if self.method in self.supervised:
            method_args = tuple([y] + list(method_args))

        # For Cholesky types, for transform from kernel to Cholesky space
        Gs = []
        Ts = []
        As = []
        for K in Ks:
            lr_model = self.method_class(rank=self.rank, **self.method_init_args)
            lr_model.fit(K, *method_args)

            # Load active set and approximation
            active = lr_model.active_set_
            G      = lr_model.G

            # Fit Nystrom method required for transform
            nystrom = Nystrom()
            nystrom.fit(K, inxs=active)
            K_SS_i = nystrom.K_SS_i
            K_XS   = nystrom.K_XS

            T = inv(G.T.dot(G)).dot(G.T).dot(K_XS).dot(K_SS_i)
            Ts.append(T)

            Gs.append(G)
            As.append(active)

        # Store feature space, transforms and active sets
        self.Gs = Gs
        self.Ts = Ts
        self.As = self.active_set_ = As

        # Fit regression model
        X = hstack(self.Gs)
        self.reg_model.fit(X, y)
        self.trained = True


    def predict(self, Xs):
        """Predict responses for test samples.

        :param Xs: (``list``) of (``numpy.ndarray``) Input space representation for each kernel in ``self.Ks``.

        :return: (``numpy.ndarray``) Vector of prediction of regression targets.
        """
        assert self.trained
        Gs = []
        for K, X, Trn, active in zip(self.Ks, Xs, self.Ts, self.As,):
            K_ST = K(K.data[active], X)
            G    = Trn.dot(K_ST).T
            Gs.append(G)

        XT = hstack(Gs)
        return self.reg_model.predict(X=XT).ravel()

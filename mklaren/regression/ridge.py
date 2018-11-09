"""
A Ridge regression in the feature space induced by multiple kernel low-rank approximations.

Implemented for Nystrom and Cholesky-type decompositions.
"""


from ..mkl.align import Align, AlignLowRank
from ..mkl.alignf import Alignf, AlignfLowRank
from ..mkl.l2krr import L2KRR, L2KRRlowRank
from ..mkl.uniform import UniformAlignment, UniformAlignmentLowRank

from ..kernel.kernel import center_kernel_low_rank
from ..kernel.kinterface import Kinterface
from ..projection.icd import ICD
from ..projection.csi import CSI
from ..projection.nystrom import Nystrom

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from numpy import array, hstack, sqrt, where, isnan, zeros, cumsum
from numpy import hstack, array, absolute
from numpy.linalg import inv, norm, lstsq


class RidgeMKL:
    """A MKL model in a transductive setting (test points are presented at training time).

    """

    mkls = {
        "align": Align,
        "alignf": Alignf,
        "alignfc": Alignf,
        "l2krr": L2KRR,
        "uniform": UniformAlignment,
    }

    mkls_low_rank = {
        "align": AlignLowRank,
        "alignf": AlignfLowRank,
        "alignfc": AlignfLowRank,
        "l2krr": L2KRRlowRank,
        "uniform": UniformAlignmentLowRank,
    }

    #  alignf expects kernels to be centered
    centered   = {"alignf", "alignfc"}
    supervised = {"align", "alignf", "alignfc", "l2krr"}

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
        self.mu         = None
        self.trained    = False


    def fit(self, Ks, y, holdout=()):
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
        self.mu = self.mkl_model.mu

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
    METHODS = ["csi", "icd", "nystrom"]
    CHOLESKY = "chol"
    NYSTROM  = "nystrom"
    methods  = {
        CHOLESKY: {"icd": ICD,
                   "csi": CSI,},
        NYSTROM: {"nystrom": Nystrom, }
    }
    supervised = ["csi"]

    def __init__(self, method_init_args=None,
                 method="icd", lbd=0, rank=10, normalize=False):
        """
        Initialize object.

        :param method: (``string``) "icd", "csi", or "nystrom", low-rank method to be used.

        :param rank: (``int``) Maximal decomposition rank.

        :param normalize: (``bool``) Normalize data in Ridge model.

        :param sum_kernels: (``bool``) Sum kernels prior to fitting. This will invoke computation of full kernel matrix.

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
        self.mu         = None
        self.lr_models  = list()
        self.lbd = lbd
        self.normalize = normalize
        self.reg_model  = Ridge(alpha=self.lbd, normalize=normalize)
        self.beta = None
        self.model_path = None

    def fit(self, Ks, y, *method_args):
        """
        Fit multiple kernel functions.

        :param Ks: (``list``) of (``Kinterface``): List of kernel interfaces or kernel matrics.

        :param y: (``numpy.ndarray``) Class labels :math:`y_i \in {-1, 1}` or regression targets.

        :param method_args: (``dict``) Arguments to the fit method of the kernel approximation method.
        """

        # Store kernels / sum if specified
        self.Ks = Ks

        # Set y as another argument
        if self.method in self.supervised:
            method_args = tuple([y] + list(method_args))

        # For Cholesky types, for transform from kernel to Cholesky space
        Gs = []
        Ts = []
        As = []
        for K in self.Ks:
            lr_model = self.method_class(rank=self.rank, **self.method_init_args)
            lr_model.fit(K, *method_args)
            self.lr_models.append(lr_model)

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
        self.y_pred = self.reg_model.predict(X)

        # Set kernel weights (absolute values)
        self.beta = self.reg_model.coef_.ravel()
        self.mu = zeros((len(self.Ks)),)

        # Kernel weight is the corresponding portion of the beta vector
        subranks = [self.Gs[ki].shape[1] for ki in range(len(self.mu))]
        subranks = [0] + list(cumsum(subranks))
        se = zip(subranks[:-1], subranks[1:])
        for ki in range(len(self.mu)):
            self.mu[ki] = norm(self.Gs[ki].dot(self.beta[se[ki][0]:se[ki][1]]))

        # Fit regularization path
        self.fit_path(Xs=None, y=y, Ks=Ks)

    def transform(self, Xs, Ks=None):
        """Transform inputs to low-rank feature space.

        :param Xs: (``list``) of (``numpy.ndarray``) Input space representation for each kernel in ``self.Ks``.

        :param Ks: (``list``) of (``numpy.ndarray``) Values of the kernel against K[test set, training set]. Optional.

        :return: (``numpy.ndarray``) Vector of prediction of regression targets.
        """
        Gs = []
        XT = None
        if Ks is None:
            for K, X, Trn, active in zip(self.Ks, Xs, self.Ts, self.As, ):
                K_ST = K(K.data[active], X)
                G = Trn.dot(K_ST).T
                Gs.append(G.reshape((len(X), self.rank)))
            XT = hstack(Gs)
        else:
            for Kt, Trn, active in zip(Ks, self.Ts, self.As):
                G = Kt[:, active].reshape(Ks[0].shape[0], len(active)).dot(Trn.T)
                Gs.append(G.reshape(Kt.shape[0], self.rank))
            XT = hstack(Gs)
        return XT

    def predict(self, Xs, Ks=None):
        """Predict responses for test samples.

        :param Xs: (``list``) of (``numpy.ndarray``) Input space representation for each kernel in ``self.Ks``.

        :param Ks: (``list``) of (``numpy.ndarray``) Values of the kernel against K[test set, training set]. Optional.

        :return: (``numpy.ndarray``) Vector of prediction of regression targets.
        """
        XT = self.transform(Xs, Ks)
        return self.reg_model.predict(X=XT).ravel()

    def fit_path(self, Xs, y, Ks=None):
        """ Compute regularized least-squares regularization path (weights).

        :param Xs: (``list``) of (``numpy.ndarray``) Input space representation for each kernel in ``self.Ks``.

        :param y: (``numpy.ndarray``) Class labels :math:`y_i \in {-1, 1}` or regression targets.

        :param Ks: (``list``) of (``numpy.ndarray``) Values of the kernel against K[test set, training set]. Optional.
        """
        XT = self.transform(Xs, Ks)
        models = []
        for j in range(XT.shape[1]):
            model = Ridge(alpha=self.lbd,
                          normalize=self.normalize,
                          fit_intercept=False).fit(X=XT[:, :j+1], y=y)
            models.append(model)
        self.model_path = models
        return

    def predict_path(self, Xs, Ks=None):
        """ Predict values for all possible weights on the regularization path.

        :param Xs: (``list``) of (``numpy.ndarray``) Input space representation for each kernel in ``self.Ks``.

        :param Ks: (``list``) of (``numpy.ndarray``) Values of the kernel against K[test set, training set]. Optional.

        :return (``numpy.ndarray``) Predited values for each element in the path.
        """
        assert self.model_path is not None
        XT = self.transform(Xs, Ks)
        rank = len(self.model_path)
        path = zeros((Xs[0].shape[0], rank))
        for j in range(XT.shape[1]):
            yp = self.model_path[j].predict(XT[:, :j+1])
            path[:, j] = yp
        return path

    def __getitem__(self, item):
        """
        Access portions of the kernel matrix generated by the kernels.
        In case of multiple kernels, return an unweighted sum.
        :param item: (``tuple``) pair of: indices or list of indices or (``numpy.ndarray``) or (``slice``)
        to address portions of the kernel matrix.
        :return:  (``numpy.ndarray``) Value of the kernel matrix for item.
        """
        return sum([model.__getitem__(item) for model in self.lr_models])
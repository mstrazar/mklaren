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
        Ridge delve using MKL methods.

        :param lbd:
            Ridge (L2) regularization parameter.
        :param method:
            Name of MKL method, must be in self.mkls.
        :param low_rank
            Use a low-rank approximation.
        :param method_init_args:
            Kwargs for the MKL method.
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
        """
        Fit MKL models according to target vector.

        :param Ks:
            List of kernel interfaces/matrices.
        :param y:
            Target vector.
        :param holdout:
            Hold-out index set.
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
        Predict values for data on inxs.

        :param inxs:
            Indexes of samples to be predicted.
        :return:
            Vector with predictions.
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
        Use kernel low-rank approximations and ridge regression.
        Implemented for Nystrom and Cholesky-type decompositions.

        Attributes and methods assumed to be provided by models:
            Essentially:
                model.active_set_       # Active set of examples
                model.G                 # Low-rank factors

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
                 method="icd", lbd=0, max_rank=100, rank=10, normalize=False):
        """
        Initialize object.

        :param method:
            Low-rank method.
        :param max_rank:
            Maximum expected rank. Required for the caching mechanism.
        :param rank
            Initial rank to be computed.
        :param normalize:
            Normalize data in Ridge model.
        :param method_init_args:
            Initialization arguments for the low-rank approximation models.
        :param lbd:
            L2-regularization.
        :return:
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
        self.max_rank   = max_rank
        self.lr_models  = dict()
        self.reg_model  = Ridge(alpha=lbd, normalize=normalize)


    def fit(self, Ks, y, *method_args):
        """
        Fit multiple kernel functions.

        :param Ks  list, [Kinterface]:
            List of kernel interfaces.
        :param method_args:
            Arguments to the fit method.
        :return:
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
        """
        Predict labels for new samples Xs using the trained regression model.

        :param Xs:
            Data in original input space. Repeated once for each kernel.
        :return:
            Predicted responses.
        """
        assert self.trained
        Gs = []
        for K, X, Trn, active in zip(self.Ks, Xs, self.Ts, self.As,):
            K_ST = K(K.data[active], X)
            G    = Trn.dot(K_ST).T
            Gs.append(G)

        XT = hstack(Gs)
        return self.reg_model.predict(X=XT).ravel()

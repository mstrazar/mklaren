import GPy
import numpy as np
from mklaren.kernel.kernel import exponential_kernel

class FITC:
    """
    Fit a Sparse GP using the variational approximation.
    Locations of the inducing points are optimized in the continuous domain.

    Warning: this method is implemented just for comparison, to adhere with mklaren
    method names.

    Currently only works for exponential kernels. Ensure the interpretation of the
    parameters of the kernels is the same.

    """

    @staticmethod
    def gamma2lengthscale(gamma):
        return np.sqrt(1.0 / (2 * gamma))

    def __init__(self, rank=10):
        """
        Initialize model.
        :param rank: Number of inducing points.
        """
        self.rank = rank
        self.anchors_ = None
        self.model = None
        self.kernel  = None

    def fit(self, Ks, y, optimize=True, fix_kernel=False):
        """
        :param Ks: Kernel interfaces. Must contain exponential kernels.
        :param y: Output (target) values.
        :param optimize: Optimize hyperparameters. This affects the kernel object too.
        :param fix_kernel: Fix kernel hyperparameters.
        """
        assert all(map(lambda K: K.kernel == exponential_kernel, Ks))
        gammas = map(lambda K: K.kernel_args["gamma"], Ks)

        # Combine with a sum of the kernel matrix
        self.kernel = GPy.kern.RBF(1, variance=1,
                                   lengthscale=self.gamma2lengthscale(gammas[0]))
        for gm in gammas[1:]:
            self.kernel += GPy.kern.RBF(1, variance=1,
                                        lengthscale=self.gamma2lengthscale(gm))

        X = Ks[0].data
        n = X.shape[0]
        y = y.reshape((n, 1))
        self.model = GPy.models.SparseGPRegression(X, y,
                                                  num_inducing=self.rank,
                                                  kernel=self.kernel)
        if fix_kernel: self.model.kern.fix()
        if optimize: self.model.optimize()
        self.anchors_ = np.array(self.model.Z)


    def predict(self, Xs):
        """
        :param Xs: Locations where to predict the function value.
        :return: Predictions.
        """
        Xp = Xs[0]
        mean, _ = self.model.predict(Xp)
        return mean
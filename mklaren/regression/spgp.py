import warnings
try:
    import GPy
    from mklaren.kernel.kernel import exponential_kernel, matern32_gpy, matern52_gpy, periodic_gpy
except ImportError:
    warnings.warn("Install module 'GPy' to use the SPGP method.")
import numpy as np

class SPGP:
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
        try:
            GPy.__version__
        except NameError:
            raise NameError("Install module 'GPy' to use the SPGP method.")

        self.rank = rank
        self.anchors_ = None
        self.model = None
        self.kernel  = None


    def fit(self, Ks, y, optimize=True, fix_kernel=False):
        """
        :param Ks: Kernel interfaces. Must contain supported kernels.
        :param y: Output (target) values.
        :param optimize: Optimize hyperparameters. This affects the kernel object too.
        :param fix_kernel: Fix kernel hyperparameters.
        """
        X = Ks[0].data
        n, d = X.shape
        y = y.reshape((n, 1))

        kernels = []
        for Kint in Ks:
            if Kint.kernel == exponential_kernel:
                assert "gamma" in Kint.kernel_args
                kern = GPy.kern.RBF(d, lengthscale=self.gamma2lengthscale(Kint.kernel_args["gamma"]))
            elif Kint.kernel == matern32_gpy:
                kern = GPy.kern.Matern32(d, **Kint.kernel_args)
            elif Kint.kernel == matern52_gpy:
                kern = GPy.kern.Matern52(d, **Kint.kernel_args)
            elif Kint.kernel == periodic_gpy:
                # kern = GPy.kern.PeriodicExponential(d, **Kint.kernel_args)
                raise ValueError("GPy.kern.PeriodicExponential is currently "
                                 "not supported by SparseGPRegression!")
            else:
                raise ValueError("Unknown kernel: %s" % str(Kint.kernel))
            kernels.append(kern)

        self.kernel = kernels[0]
        for k in kernels[1:]: self.kernel += k

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
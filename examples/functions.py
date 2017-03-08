from mklaren.kernel.kernel import exponential_kernel, linear_kernel, poly_kernel
from scipy.stats import chi2
from random import choice
import numpy as np


class MultiKernelFunction:

    # Library of function families
    library = {
        "exp": (exponential_kernel, ("gamma", )),
        "lin": (linear_kernel, ()),
        "poly": (poly_kernel, ("p", "b"))
    }

    # Hyper parameter ranges
    values = {
        "gamma": np.logspace(-2, 2, 100),
        "p": [2, 3, 4, 5],
        "b": np.linspace(-3, 3, 100)
    }

    def __init__(self, p):
        """
        Initialize a random combination of kernels.
        :param p: Number of kernels.
        """
        self.p = p
        self.signs = []
        self.funcs = []
        self.args = []
        self.weights = []
        for pi in range(p):
            key = choice(self.library.keys())
            self.signs.append(key)

            f, args = self.library[key]
            self.funcs.append(f)

            m = map(lambda a: (a, choice(self.values[a],)), args)
            self.args.append(dict(m))

            # A random kernel weight
            self.weights.append(chi2.rvs(df=1))

    def __call__(self, X, alpha):
        """
        Return the signal encoded by the kernel function.
        :param X:
            A data matrix.
        :param alpha:
            Dual coefficients.
        :return:
        """
        n = np.array(X).shape[0]
        K = np.zeros((n, n))
        assert len(alpha) == n
        alpha = alpha.reshape((n, 1))
        for pi in range(self.p):
            K = K + self.weights[pi] * self.funcs[pi](X, X, **self.args[pi])
        return K.dot(alpha)

    def __str__(self):
        """
        Print the rule encoded by this function
        :return:
            A text representation of a random kernel function.
        """
        am = lambda pi: ", ".join("%s=%.2f" % (k, v) for k, v in self.args[pi].items())
        args = map(am,  range(self.p))

        mp = map(lambda pi: "%0.2e %s(x, x, %s)"
            % (self.weights[pi], self.signs[pi], args[pi]), range(self.p))
        txt = " + \n\t ".join(mp)
        return "y = \n\t %s" % txt
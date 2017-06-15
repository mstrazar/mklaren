import unittest
import numpy as np
import GPy
from mklaren.regression.fitc import FITC
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface
from scipy.stats import multivariate_normal as mvn


class TestFITC(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.X = np.linspace(-10, 10, self.n).reshape((self.n, 1))
        np.random.seed(42)


    def testKernGamma(self):
        for gamma in [0.1, 1.0, 2.0, 10.0]:
            k = GPy.kern.RBF(1, variance=1, lengthscale=FITC.gamma2lengthscale(gamma))
            K = k.K(self.X, self.X)
            Ki = Kinterface(data=self.X,
                            kernel=exponential_kernel,
                            kernel_args={"gamma": gamma})
            self.assertAlmostEqual(np.linalg.norm(K-Ki[:, :]), 0, places=3)


    def testFITCfit(self):
        n = self.n
        X = self.X
        noise = 1.0

        # Construct a combined kernel
        gamma_range = [0.1, 0.3, 1.0]
        Ks = [Kinterface(data=X, kernel=exponential_kernel,
                         kernel_args={"gamma": gm}) for gm in gamma_range]
        Km = Kinterface(data=X, kernel=kernel_sum,
                        kernel_args={"kernels": [exponential_kernel] * len(gamma_range),
                                     "kernels_args": map(lambda gm: {"gamma": gm}, gamma_range)})

        for seed in range(5):
            # Sample a function from a GP
            f = mvn.rvs(mean=np.zeros((n,)), cov=Km[:, :], random_state=seed)
            y = mvn.rvs(mean=f, cov=np.eye(n, n) * noise, random_state=seed)
            y = y.reshape((n, 1))

            # Fit a model
            model = FITC()
            model.fit(Ks, y)

            # Compare kernels
            self.assertAlmostEqual(np.linalg.norm(model.kernel.K(X, X) - Km[:, :]), 0, places=3)

            # Predictions
            yp = model.predict([X])
            v1 = np.var(y.ravel())
            v2 = np.var((y-yp).ravel())
            self.assertTrue(v2 < v1)
import unittest
import numpy as np
import GPy
from mklaren.regression.spgp import SPGP
from mklaren.kernel.kernel import exponential_kernel, kernel_sum, matern52_gpy, matern32_gpy, periodic_gpy
from mklaren.kernel.kinterface import Kinterface
from scipy.stats import multivariate_normal as mvn


class TestSPGP(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.X = np.linspace(-10, 10, self.n).reshape((self.n, 1))
        np.random.seed(42)


    def testKernGamma(self):
        for gamma in [0.1, 1.0, 2.0, 10.0]:
            k = GPy.kern.RBF(1, variance=1, lengthscale=SPGP.gamma2lengthscale(gamma))
            K = k.K(self.X, self.X)
            Ki = Kinterface(data=self.X,
                            kernel=exponential_kernel,
                            kernel_args={"gamma": gamma})
            self.assertAlmostEqual(np.linalg.norm(K-Ki[:, :]), 0, places=3)


    def testSPGPfit(self):
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
            model = SPGP()
            model.fit(Ks, y, optimize=False, fix_kernel=False)

            # Compare kernels
            self.assertAlmostEqual(np.linalg.norm(model.kernel.K(X, X) - Km[:, :]), 0, places=3)

            # Predictions
            yp = model.predict([X])
            v1 = np.var(y.ravel())
            v2 = np.var((y-yp).ravel())
            self.assertTrue(v2 < v1)

            # Fixed model
            model_fix = SPGP()
            model_fix.fit(Ks, y, optimize=False, fix_kernel=True)
            ypf = model_fix.predict([X])
            v3 = np.var((y - ypf).ravel())
            self.assertTrue(v3 < v1)


    def testAllKernels(self):
        X = self.X
        y = np.random.rand(X.shape[0], 1)

        Ks = [Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.1}),
              Kinterface(data=X, kernel=matern32_gpy, kernel_args={"lengthscale": 3.0}),
              Kinterface(data=X, kernel=matern52_gpy, kernel_args={"lengthscale": 5.0}),
              # Kinterface(data=X, kernel=periodic_gpy, kernel_args={"lengthscale": 5.0, "period": 4.0}),
              ]
        Km = sum([K[:, :] for K in Ks])

        kern = GPy.kern.RBF(1, lengthscale=SPGP.gamma2lengthscale(0.1)) \
               + GPy.kern.Matern32(1, lengthscale=3) \
               + GPy.kern.Matern52(1, lengthscale=5)
               # + GPy.kern.PeriodicExponential(1, lengthscale=5, period=4)

        Ky = kern.K(X, X)
        self.assertAlmostEqual(np.linalg.norm(Ky - Km[:, :]), 0, places=3)

        model = SPGP()
        model.fit(Ks, y, optimize=True, fix_kernel=True)
        yp = model.predict([X])
        v1 = np.var(y.ravel())
        v2 = np.var((y - yp).ravel())
        self.assertTrue(v2 < v1)


if __name__ == "__main__":
    unittest.main()
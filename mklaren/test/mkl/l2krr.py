import unittest
import numpy as np
from mklaren.mkl.l2krr import L2KRR
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface



class TestL2KRR(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.m = 3
        self.gamma_range = np.logspace(-1, 1, 5)
        self.lbd_range = [0, 1, 100, 1000]
        self.X = np.random.rand(self.n, self.m)
        self.Ks = [Kinterface(data=self.X,
                              kernel=exponential_kernel,
                              kernel_args={"gamma": g}) for g in self.gamma_range]
        self.trials = 5

    def testFitting(self):

        for t in range(self.trials):
            alpha = np.random.randn(self.n, 1)
            mu0 = np.random.randn(len(self.Ks), 1)
            K0 = sum([w * K[:, :] for K, w in zip(self.Ks, mu0)])
            y = K0.dot(alpha)
            y = y - y.mean()

            evars = np.zeros((len(self.lbd_range),))
            for li, lbd in enumerate(self.lbd_range):
                model = L2KRR(lbd2=10, lbd=lbd)
                model.fit(self.Ks, y)
                yp = model.Kappa.dot(model.alpha)
                expl_var = (np.var(y) - np.var(y - yp)) / np.var(y)
                evars[li] = expl_var
                if model.lbd == 0:
                    self.assertTrue(expl_var > 0.9)

            self.assertTrue(np.all(evars[0:-1] > evars[1:]))


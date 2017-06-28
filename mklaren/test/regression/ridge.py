import unittest
import numpy as np
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.regression.ridge import RidgeMKL


class TestRidge(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.m = 2
        self.gamma_range = np.logspace(-1, 1, 5)
        self.trials = 5

    def testPrediction(self):
        for t in range(self.trials):
            X = np.random.rand(self.n, self.m)
            tr = np.arange(self.n/2).astype(int)    # necessarily int 1D array
            te = np.arange(self.n/2, self.n).astype(int)
            Ks = [Kinterface(data=X,
                                  kernel=exponential_kernel,
                                  kernel_args={"gamma": g}) for g in self.gamma_range]

            inxs = np.random.choice(tr.ravel(), size=3)
            alpha = np.zeros((self.n, 1))
            alpha[inxs] = np.random.randn(3, 1)
            mu0 = np.random.randn(len(Ks), 1)
            K0 = sum([w * K[:, :] for K, w in zip(Ks, mu0)])
            y = K0.dot(alpha).ravel()
            y = y - y.mean()    # y necessarily 1D array

            for method in RidgeMKL.mkls.keys():
                model = RidgeMKL(method=method)
                model.fit(Ks, y, holdout=te)
                yp = model.predict(te)
                expl_var = (np.var(y[te]) - np.var(y[te] - yp)) / np.var(y[te])
                self.assertGreater(expl_var, 0.5)
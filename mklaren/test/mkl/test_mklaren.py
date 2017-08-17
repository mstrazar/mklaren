import unittest
import numpy as np
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren

class TestMklaren(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.m = 10
        self.trials = 5

    def testPredictionKernPrecomp(self):
        for t in range(self.trials):
            X = np.random.rand(self.n, self.m)
            Ks =  [Kinterface(kernel=exponential_kernel, data=X, kernel_args={"gamma": 0.1}),
                   Kinterface(kernel=exponential_kernel, data=X, kernel_args={"gamma": 0.2}),]
            Ls = [K[:, :] for K in Ks]
            y = X[:, :3].sum(axis=1)
            y = y - y.mean()

            X_te = np.random.rand(10, self.m)
            Ls_te = [K(X_te, X) for K in Ks]

            # Kinterface model
            model0 = Mklaren(lbd=0.01, rank=5)
            model0.fit(Ks, y)
            y0 = model0.predict([X, X])
            yp0 = model0.predict([X_te, X_te])

            # Kernel matrix model
            model1 = Mklaren(lbd=0.01, rank=5)
            model1.fit(Ls, y)
            y1 = model0.predict(Xs=None, Ks=Ls)
            yp1 = model0.predict(Xs=None, Ks=Ls_te)

            self.assertAlmostEqual(np.linalg.norm(y0-y1), 0, places=3)
            self.assertAlmostEqual(np.linalg.norm(yp0-yp1), 0, places=3)
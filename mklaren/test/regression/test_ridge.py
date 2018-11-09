import unittest
import numpy as np
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.regression.ridge import RidgeMKL, RidgeLowRank


class TestRidge(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.m = 2
        self.gamma_range = np.logspace(-1, 1, 5)
        self.trials = 5

    def testPrediction(self):
        np.random.seed(42)
        for t in range(self.trials):
            X = np.random.rand(self.n, self.m)
            tr = np.arange(self.n/2).astype(int)    # necessarily int 1D array
            te = np.arange(self.n/2, self.n).astype(int)
            Ks = [Kinterface(data=X,
                                  kernel=exponential_kernel,
                                  kernel_args={"gamma": g}) for g in self.gamma_range]

            inxs = np.random.choice(tr.ravel(), size=self.n/3)
            alpha = np.zeros((self.n, 1))
            alpha[inxs] = np.random.randn(len(inxs), 1)
            mu0 = np.random.randn(len(Ks), 1)
            K0 = sum([w * K[:, :] for K, w in zip(Ks, mu0)])
            y = K0.dot(alpha).ravel()
            y = y - y.mean()    # y necessarily 1D array
            y += np.random.randn(len(K0), 1).ravel() * 0.001

            for method in RidgeMKL.mkls.keys():

                model = RidgeMKL(method=method)
                model.fit(Ks, y, holdout=te)
                yp = model.predict(te)
                expl_var = (np.var(y[te]) - np.var(y[te] - yp)) / np.var(y[te])
                self.assertGreater(expl_var, 0.5)


class TestRidgeLowRank(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.m = 10
        self.gamma_range = np.logspace(-1, 1, 5)
        self.trials = 5

    def testPrediction(self):
        for t in range(self.trials):
            X = np.random.rand(self.n, self.m)
            te = np.arange(5*self.n/6, self.n).astype(int)
            Vs = [X[:, i].reshape(self.n, 1) for i in range(self.m)]
            y = X[:, :3].sum(axis=1)
            y = y - y.mean()
            for method in RidgeMKL.mkls_low_rank.keys():
                model = RidgeMKL(method=method, low_rank=True, lbd=0.01)
                model.fit(Vs, y, holdout=te)
                yp = model.predict(te)
                expl_var = (np.var(y[te]) - np.var(y[te] - yp)) / np.var(y[te])
                method, expl_var, model.mu.ravel()
                self.assertGreater(expl_var, 0.1)

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
            for method in ["icd", "csi", "nystrom"]:
                # Kinterface model
                model0 = RidgeLowRank(method=method, lbd=0.01)
                model0.fit(Ks, y)
                y0 = model0.predict([X, X])
                yp0 = model0.predict([X_te, X_te])

                # Kernel matrix model
                model1 = RidgeLowRank(method=method, lbd=0.01)
                model1.fit(Ls, y)
                y1 = model0.predict(Xs=None, Ks=Ls)
                yp1 = model0.predict(Xs=None, Ks=Ls_te)

                self.assertAlmostEqual(np.linalg.norm(y0-y1), 0, places=3)
                self.assertAlmostEqual(np.linalg.norm(yp0-yp1), 0, places=3)

    def testPredictPath(self):
        """ Test consistency of predicted path. """
        X = np.random.rand(self.n, self.m)
        Ks = [Kinterface(kernel=exponential_kernel, data=X, kernel_args={"gamma": 0.1}),
              Kinterface(kernel=exponential_kernel, data=X, kernel_args={"gamma": 0.2}), ]
        y = X[:, :3].sum(axis=1)
        y = y - y.mean()
        for method in ["icd", "csi", "nystrom"]:
            model0 = RidgeLowRank(method=method, lbd=0.0, rank=20)
            model0.fit(Ks, y)
            A = model0.transform([X, X])
            ypath = model0.predict_path([X, X])
            norms = np.linalg.norm(ypath, axis=0)
            assert np.all(norms[1:] > norms[:-1])
            assert np.linalg.norm(A.T.dot(y - ypath[:, -1])) < 1e-5





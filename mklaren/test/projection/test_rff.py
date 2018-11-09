import unittest
import numpy as np
from mklaren.kernel.kernel import exponential_kernel
from mklaren.projection.rff import RFF_KMP, RFF, RFF_NS, RFF_TYPES


class TestRFF(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.p = 10
        self.rank_range = [3, 100]
        self.X = np.random.rand(self.n, self.p)
        self.gamma_range = np.logspace(-3, -1, 5)
        self.lbd_range = np.logspace(-2, 2, 5)

        # Generate a random signal by employing one of the kernels
        K = exponential_kernel(self.X, self.X, gamma=self.gamma_range[1])
        self.alpha = np.random.rand(self.n, 1)
        self.y = K.dot(self.alpha).ravel()

    def testReconstruction(self):
        """
        Fitted features must match transformed ones.
        """
        model = RFF_KMP(delta=10, rank=30, gamma_range=self.gamma_range, lbd=0)
        model.fit(self.X, self.y)
        G = model.G
        Gt = model.transform(self.X)
        self.assertAlmostEqual(np.linalg.norm(G - Gt), 0, places=3)

    def testFitting(self):
        """
        Reconstruction must improve with increasing rank and
        random initial conditions
        """
        errors = np.zeros((len(self.rank_range),))
        for typ in RFF_TYPES:
            for ri, rank in enumerate(self.rank_range):
                model = RFF_KMP(delta=10, rank=rank,
                                gamma_range=self.gamma_range, random_state=42,
                                lbd=0, typ=typ)
                model.fit(self.X, self.y)
                yp = model.predict(self.X)
                errors[ri] = np.sum((self.y.ravel() - yp.ravel()) ** 2)
        self.assertTrue(np.all(errors[:-1] > errors[1:]))

    def testRegularization(self):
        """
        Reconstruction must improve with increasing rank and
        random initial conditions
        """
        errors = np.zeros((len(self.lbd_range),))
        beta_norms = np.zeros((len(self.lbd_range),))

        for typ in RFF_TYPES:
            for li, lbd in enumerate(self.lbd_range):
                model = RFF_KMP(delta=10, rank=30,
                                gamma_range=self.gamma_range, random_state=42,
                                lbd=lbd, typ=typ)
                model.fit(self.X, self.y)
                yp = model.predict(self.X)
                errors[li] = np.sum((self.y.ravel() - yp.ravel()) ** 2)
                beta_norms[li] = np.linalg.norm(model.beta)

            # Error grows with lambda
            # Vector norm drops
            self.assertTrue(np.all(errors[:-1] < errors[1:]))
            self.assertTrue(np.all(beta_norms[:-1] > beta_norms[1:]))

    def testConvergence(self):
        """
        Make sure both definitions of gamma are equivalent.
        :return:
        """

        for g in self.gamma_range:
            K = exponential_kernel(self.X, self.X, gamma=g)[:, :]
            y = K.dot(np.ones((self.n, 1)))

            model = RFF_KMP(delta=0, rank=100,
                            gamma_range=[g], random_state=42, lbd=0.0, normalize=False)
            model.fit(self.X, y)

            errors = np.zeros((len(self.rank_range),))
            for ri, rank in enumerate(self.rank_range):
                Gt = model.transform(X=self.X)[:, :rank]
                Kt = Gt.dot(Gt.T)
                errors[ri] = np.linalg.norm(K - Kt)

            self.assertTrue(np.all(errors[:-1] > errors[1:]))

    def testReconstructionRFF(self):
        """ Test approximation of the exponential kernel. """
        np.random.seed(42)
        n = 100
        d = 1
        X = np.linspace(-10, 10, n).reshape((n, d))
        gamma_range = np.logspace(-3, 2, 6)
        rank = 3000
        for gam in gamma_range:
            K = exponential_kernel(X, X, gamma=gam)
            model = RFF(d=d, n_components=rank, gamma=gam, random_state=42)
            model.fit()
            G = model.transform(X)
            L = G.dot(G.T)
            assert G.shape == (n, rank)
            assert np.linalg.norm(L - K) < 10

    def testNonstat(self):
        """ Test non-stationary model. """
        np.random.seed(42)
        n = 100
        d = 1
        X = np.linspace(-10, 10, n).reshape((n, d))
        rank = 3000

        # Compare results using two different models
        model = RFF_NS(d=d, n_components=rank,
                       kwargs1={"gamma": 0.1},
                       kwargs2={"gamma": 0.01})
        model.fit()
        G = model.transform(X)
        assert G.shape == (n, rank)

if __name__ == "__main__":
    unittest.main()

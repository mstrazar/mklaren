import numpy as np
import unittest
from mklaren.projection.nystrom import Nystrom
from mklaren.kernel.kernel import poly_kernel

class TestNystrom(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.p = 100
        self.X = np.random.rand(self.n, self.p)


    def testDeterministicDecrease(self):
        """
        Test expected reconstruction properties of the Nystrom method.
        """
        for d in range(1, 6):
            K = poly_kernel(self.X, self.X, degree=d)
            model = Nystrom(rank=self.n, random_state=42)
            model.fit(K)

            errors = np.zeros((self.n, ))
            for i in range(self.n):
                Ki = model.G[:, :i+1].dot(model.G[:, :i+1].T)
                errors[i] = np.linalg.norm(K-Ki)

            self.assertTrue(np.all(errors[:-1] > errors[1:]))
            self.assertAlmostEqual(errors[-1], 0, delta=3)


    def testLeverage(self):
        """
        Assert the leverage scores performs a better low rank approximation with incraesing number of
        columns.
        :return:
        """
        K = poly_kernel(self.X, self.X, degree=2)
        rank_range = [10, 20, 30, 50]
        repeats = 10

        errors_lev = np.zeros((repeats, len(rank_range),))
        errors_rand = np.zeros((repeats, len(rank_range),))

        for j in xrange(repeats):
            self.X = np.random.rand(self.n, self.p)
            for i, rank in enumerate(rank_range):
                model_lev = Nystrom(rank=rank, random_state=j, lbd=1)
                model_lev.fit(K)

                model_rand = Nystrom(rank=rank, random_state=j, lbd=0)
                model_rand.fit(K)

                Li = model_lev.G.dot(model_lev.G.T)
                Ri = model_rand.G.dot(model_rand.G.T)
                errors_lev[j, i] = np.linalg.norm(K - Li)
                errors_rand[j, i] = np.linalg.norm(K - Ri)
            self.assertTrue(np.all(errors_lev[j, :-1] > errors_lev[j, 1:]))

        lev_win = np.sum(errors_lev < errors_rand)
        rand_win = np.sum(errors_lev > errors_rand)
        print("Leverage win: %d, random win: %d" % (lev_win, rand_win))
        self.assertTrue(lev_win > rand_win)


if __name__ == "__main__":
    unittest.main()

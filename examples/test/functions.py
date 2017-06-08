import unittest

import numpy as np

import examples.archive.functions as func


class TestFunctions(unittest.TestCase):

    def setUp(self):
        self.repeats = 10
        self.n = 100
        self.p = 30
        self.mf = func.MultiKernelFunction(5, row_normalize=True)
        np.random.seed(42)


    def testEigenvalues(self):
        """
        Test positive semidefiniteness.
        :return:
        """
        for repeat in range(self.repeats):
            X = np.random.rand(self.n, self.p)
            Ks, names = func.data_kernels(X, mf=self.mf,
                                          row_normalize=True, noise=0.01)
            for n, K in zip(names, Ks):
                v, _ = np.linalg.eig(K)
                vr = np.round(np.real(v), 3)
                self.assertTrue(all(vr >= 0))

    def testRank(self):
        """
        Test matrix rank.
        :return:
        """
        for repeat in range(self.repeats):
            X = np.random.rand(self.n, self.p)
            Ks, names = func.data_kernels(X, mf=self.mf,
                                          row_normalize=True, noise=0.01)
            for n, K in zip(names, Ks):
                rnk = np.linalg.matrix_rank(K)
                self.assertTrue(rnk == self.n)

    def testDifferent(self):
        """
        Assert all matrices are different
        :return:
        """
        for repeat in range(self.repeats):
            X = np.random.rand(self.n, self.p)
            listing = self.mf.kernel_generator()
            _, _, args = zip(*listing)
            Ks, names = func.data_kernels(X, mf=self.mf,
                                          row_normalize=True, noise=0.01)
            data = dict()
            for n, K, ka in zip(names, Ks, args):
                for ky, (L, la) in data.items():
                    check = np.round(np.linalg.norm(L-K), 3) > 0
                    if not check:
                        print("%s %s" % (n, ka))
                        print("equals")
                        print("%s %s" % (ky, la))
                    self.assertTrue(check)
                data[n] = (K, ka)

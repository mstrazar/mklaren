import unittest

import numpy as np

import examples.features as ft


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.n = 103
        self.ps = [3, 4, 5]
        self.degrees = [2, 3, 4, 5]
        self.repeats = 1

    def testDotProduct(self):
        for p, d in ft.product(self.ps, self.degrees):
            X = np.ones((self.n, p)).astype(int)
            pf = ft.Features(degree=d)
            Z = pf.fit_transform(X)
            P1 = sum([(X.dot(X.T))**e for e in range(d+1)])
            P2 = Z.dot(Z.T)
            self.assertAlmostEqual(np.linalg.norm(P1-P2), 0, delta=5)


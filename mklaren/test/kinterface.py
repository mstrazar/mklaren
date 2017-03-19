import unittest
import numpy as np
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import poly_kernel, kernel_row_normalize


class TestKinterface(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n = 30
        self.m = 20
        self.p = 10
        self.X = np.random.rand(self.n, self.p)
        self.Y = np.random.rand(self.m, self.p)


    def testKinterface(self):
        pass

    def testGetItem(self):
        Kp = poly_kernel(self.X, self.X, degree=2)
        Ki = Kinterface(data=self.X, kernel=poly_kernel, kernel_args={"degree": 2})
        self.assertAlmostEquals(np.linalg.norm(Ki[:, :] - Kp), 0, delta=3)

    def testCall(self):
        Kp = poly_kernel(self.X, self.X, degree=2)
        Ki = Kinterface(data=self.X, kernel=poly_kernel, kernel_args={"degree": 2})
        self.assertAlmostEquals(np.linalg.norm(Ki(self.X, self.X) - Kp), 0, delta=3)

    def testRowNorm(self):
        Kp = poly_kernel(self.X, self.X, degree=2)
        Kr = kernel_row_normalize(Kp)
        Ki = Kinterface(data=self.X, kernel=poly_kernel, kernel_args={"degree": 2},
                        row_normalize=True)
        self.assertAlmostEquals(np.linalg.norm(Ki.diag().ravel() - np.ones((self.n, ))), 0, delta=3)
        self.assertAlmostEquals(np.linalg.norm(Ki(self.X, self.X) - Kr), 0, delta=3)
        self.assertAlmostEquals(np.linalg.norm(Ki[:, :] - Kr), 0, delta=3)


    def testCallOther(self):
        Kp = poly_kernel(self.X, self.Y, degree=2)
        Ki = Kinterface(data=self.X, kernel=poly_kernel, kernel_args={"degree": 2},
                        row_normalize=False)
        Kr = Ki(self.X, self.Y)
        self.assertAlmostEquals(np.linalg.norm(Kp - Kr), 0, delta=3)


    def testCallOtherNorm(self):
        Ki = Kinterface(data=self.X, kernel=poly_kernel, kernel_args={"degree": 2},
                        row_normalize=True)
        Kr = Ki(self.X, self.Y)
        self.assertTrue(np.all(Kr < 1))

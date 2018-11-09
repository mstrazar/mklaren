import numpy as np
import unittest
from mklaren.projection.icd import ICD
from mklaren.kernel.kernel import poly_kernel
from mklaren.kernel.kinterface import Kinterface

class TestICD(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.p = 100
        self.X = np.random.rand(self.n, self.p)


    def testPoly(self):
        """
        Test expected reconstruction properties of the ICD.
        """
        for d in range(1, 6):
            K = poly_kernel(self.X, self.X, degree=d)
            model = ICD(rank=self.n)
            model.fit(K)

            errors = np.zeros((self.n, ))
            for i in range(self.n):
                Ki = model.G[:, :i+1].dot(model.G[:, :i+1].T)
                errors[i] = np.linalg.norm(K-Ki)

            self.assertTrue(np.all(errors[:-1] > errors[1:]))
            self.assertAlmostEqual(errors[-1], 0, delta=3)


    def testPolySum(self):
        """
        Test expected reconstruction properties of the ICD.
        Kernels are iteratively summed.
        """
        K = np.zeros((self.n, self.n))
        for d in range(1, 6):
            K += Kinterface(data=self.X, kernel=poly_kernel,
                            kernel_args={"degree": d},
                            row_normalize=True)[:, :]
            model = ICD(rank=self.n)
            model.fit(K)

            errors = np.zeros((self.n, ))
            for i in range(self.n):
                Ki = model.G[:, :i+1].dot(model.G[:, :i+1].T)
                errors[i] = np.linalg.norm(K-Ki)

            self.assertTrue(np.all(errors[:-1] > errors[1:]))
            self.assertAlmostEqual(errors[-1], 0, delta=3)


if __name__ == "__main__":
    unittest.main()

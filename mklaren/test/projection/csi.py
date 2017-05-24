import numpy as np
import unittest
from mklaren.projection.csi import CSI
from mklaren.kernel.kernel import poly_kernel

class TestCSI(unittest.TestCase):

    def setUp(self):
        self.n = 100
        self.p = 100
        self.X = np.random.rand(self.n, self.p)


    def testPoly(self):
        """
        Test expected reconstruction properties of the ICD.
        """
        delta = 5
        rank = self.n
        for d in range(1, 6):
            K = poly_kernel(self.X, self.X, degree=d)
            y = np.random.rand(self.n, 1)
            model = CSI(rank=rank, delta=delta, kappa=0.1)
            model.fit(K, y)

            errors = np.zeros((rank, ))
            for i in range(rank):
                Ki = model.G[:, :i+1].dot(model.G[:, :i+1].T)
                errors[i] = np.linalg.norm(K - Ki)

            self.assertAlmostEqual(errors[-1], 0, places=3)
            self.assertTrue(np.all(errors[:-1] >= errors[1:]))

if __name__ == '__main__':
    unittest.main()
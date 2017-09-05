import unittest
from mklaren.kernel.string_kernel import *
from mklaren.kernel.string_util import *
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
import scipy.stats as st


class TestString(unittest.TestCase):

    def setUp(self):
        X, y = generate_data(N=100, L=100, p=0.5, motif="TGTG", mean=0, var=3, seed=42)
        self.Xa = np.array(X)
        self.y = y

        self.Ks = [Kinterface(kernel=string_kernel, data=self.Xa, kernel_args={"mode": SPECTRUM}),
                   Kinterface(kernel=string_kernel, data=self.Xa, kernel_args={"mode": SPECTRUM_MISMATCH})]


    def testMklarenFit(self):
        model = Mklaren(rank=5)
        model.fit(self.Ks, self.y)
        yp = model.predict([self.Xa] * len(self.Ks))
        c, p = st.spearmanr(yp, self.y)
        self.assertGreater(c, 0)
        self.assertLess(p, 0.05)


    def testCSIFit(self):
        Ks = [Kinterface(kernel=string_kernel, data=self.Xa, kernel_args={"mode": SPECTRUM})]
        model = RidgeLowRank(rank=5, method="csi", method_init_args={"delta": 5}, lbd=0.01)
        model.fit(Ks, self.y)
        yp = model.predict([self.Xa] * len(Ks))
        c, p = st.spearmanr(yp, self.y)
        self.assertGreater(c, 0)
        self.assertLess(p, 0.05)


    def testMklarenPredict(self):
        X_tr = self.Xa[:50]
        X_te = self.Xa[50:]
        y_tr = self.y[:50]
        y_te = self.y[50:]

        Ks = [Kinterface(kernel=string_kernel, data=X_tr, kernel_args={"mode": SPECTRUM}),
              Kinterface(kernel=string_kernel, data=X_tr, kernel_args={"mode": SPECTRUM_MISMATCH})]

        model = Mklaren(rank=10)
        model.fit(Ks, y_tr)
        yp = model.predict([X_te] * len(Ks))

        c, p = st.spearmanr(yp, y_te)
        self.assertGreater(c, 0)
        self.assertLess(p, 0.05)
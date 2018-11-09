import numpy as np
import unittest
from mklaren.mkl.kmp import KMP
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import exponential_kernel
from scipy.stats import multivariate_normal as mvn


class TestKMP(unittest.TestCase):

    def test_least_squares_sol(self):
        np.random.seed(1)
        n = 100
        rank = 20
        delta = 5
        X = np.linspace(-10, 10, n).reshape((n, 1))
        Ks = [
            Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.6}),
            Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.1}),
            ]
        Kt = 1.0 + Ks[0][:, :] + 0.0 * Ks[1][:, :]
        y = mvn.rvs(mean=np.zeros(n,), cov=Kt).reshape((n, 1))
        y = y - y.mean()
        model = KMP(rank=rank, delta=delta, lbd=0)
        model.fit(Ks, y)
        yp = model.predict([X, X])
        assert np.linalg.norm(yp.T.dot(y.ravel() - yp)) < 1e-2

    def test_high_rank(self):
        np.random.seed(2)
        n = 100
        rank = 200
        delta = 5
        X = np.linspace(-10, 10, n).reshape((n, 1))
        Ks = [
            Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.6}),
            Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.1}),
            ]
        Kt = 1.0 + Ks[0][:, :] + 0.0 * Ks[1][:, :]
        y = mvn.rvs(mean=np.zeros(n,), cov=Kt).reshape((n, 1))
        y = y - y.mean()
        model = KMP(rank=rank, delta=delta, lbd=0)
        model.fit(Ks, y)
        assert model.rank <= n
        yp = model.predict([X, X])
        assert np.linalg.norm(yp.T.dot(y.ravel() - yp)) < 1e-2

    def test_high_delta(self):
        np.random.seed(3)
        n = 100
        rank = 20
        delta = 100
        X = np.linspace(-10, 10, n).reshape((n, 1))
        Ks = [
            Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.6}),
            Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.1}),
            ]
        Kt = 1.0 + Ks[0][:, :] + 0.0 * Ks[1][:, :]
        y = mvn.rvs(mean=np.zeros(n,), cov=Kt).reshape((n, 1))
        y = y - y.mean()
        model = KMP(rank=rank, delta=delta, lbd=0)
        model.fit(Ks, y)
        assert model.rank <= n
        yp = model.predict([X, X])
        assert np.linalg.norm(yp.T.dot(y.ravel() - yp)) < 1e-2

    def test_coef_path(self):
        """ Assert least squares solution is valid at each step. """
        n = 100
        rank = 20
        delta = 5
        X = np.linspace(-10, 10, n).reshape((n, 1))
        Ks = [
            Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.6}),
            Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.1}),
            ]
        Kt = 1.0 + Ks[0][:, :] + 0.0 * Ks[1][:, :]
        y = mvn.rvs(mean=np.zeros(n,), cov=Kt).reshape((n, 1))
        y = y - y.mean()
        model = KMP(rank=rank, delta=delta, lbd=0)
        model.fit(Ks, y)
        ypath = model.predict_path([X, X])
        for i in range(model.rank):
            assert np.linalg.norm(ypath[:, i].T.dot(y.ravel() - ypath[:, i])) < 1e-3

    def test_bias(self):
        """ Assert least squares solution is valid at each step. """
        n = 100
        rank = 20
        delta = 5
        bias = 20
        X = np.linspace(-10, 10, n).reshape((n, 1))
        Ks = [
            Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.6}),
            Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.1}),
        ]
        Kt = 1.0 + Ks[0][:, :] + 0.0 * Ks[1][:, :]
        y = mvn.rvs(mean=np.zeros(n, ), cov=Kt).reshape((n, 1))
        y = y + bias
        model = KMP(rank=rank, delta=delta, lbd=0)
        model.fit(Ks, y)
        ypath = model.predict_path([X, X])
        for i in range(model.rank):
            yp = ypath[:, i] - model.bias
            yu = y.ravel() - model.bias
            assert np.linalg.norm(yp.T.dot(yu - yp)) < 1e-3
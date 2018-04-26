import numpy as np
from mklaren.kernel.kernel import exponential_kernel
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt


def exponential_density(rank, d, sigma=None, gamma=1):
    """ Return samples for the exponential kernel. """
    if sigma is None:
        sigma = np.sqrt(2 * d * gamma)
    return mvn.rvs(mean=np.zeros(d,), cov=sigma**2 * np.eye(d, d), size=rank).reshape((rank, d))


class RFF:
    """
    Generalized Random Fourier Features Sampler.
    """

    def __init__(self, d, rank=10, density=exponential_density, **kwargs):
        self.d = d
        self.rank = rank
        self.W = None
        self.b = None
        self.density = density
        self.kwargs = kwargs

    def fit(self):
        """ Sample random directions for a given kernel. """
        self.W = self.density(self.rank, self.d, **self.kwargs)
        # self.b = np.random.rand(self.rank, 1) * np.pi * 2

    def transform(self, X):
        """ Map X to approximation of the kernel. """
        # Rahimi & Recht 2007 - real part only (rank)
        # A = X.dot(self.W.T) + self.b.T
        # return np.sqrt(2.0 / self.rank) * np.cos(A)

        # Ton et. al 2018 - real + complex (2k)
        A = X.dot(self.W.T)
        return 1.0 / np.sqrt(self.rank) * np.hstack((np.cos(A), np.sin(A)))


class RFF_NS:
    """
    Generalized Random Fourier Features Sampler for non-stationary kernel.
    """

    def __init__(self, d, rank=10,
                 density1=exponential_density, kwargs1={},
                 density2=exponential_density, kwargs2={}):
        self.d = d
        self.rank = rank
        self.W1 = None
        self.W2 = None
        self.density1 = density1
        self.density2 = density2
        self.kwargs1 = kwargs1
        self.kwargs2 = kwargs2

    def fit(self):
        """ Sample random directions for a given kernel. """
        self.W1 = self.density1(self.rank, self.d, **self.kwargs1)
        self.W2 = self.density2(self.rank, self.d, **self.kwargs2)

    def transform(self, X):
        """ Map X to approximation of the kernel. """
        # Rahimi & Recht 2007 - real part only (rank)
        # A = X.dot(self.W.T) + self.b.T
        # return np.sqrt(2.0 / self.rank) * np.cos(A)

        # Ton et. al 2018 - real + complex (2k)
        A = X.dot(self.W1.T)
        B = X.dot(self.W2.T)
        return np.sqrt(1.0 / (4.0 * self.rank)) * np.hstack((np.cos(A) + np.cos(B),
                                                             np.sin(A) + np.sin(B)))


def test_exponential():
    """ Test approximation of the exponential kernel. """
    np.random.seed(42)
    n = 100
    d = 1
    X = np.linspace(-10, 10, n).reshape((n, d))
    gamma_range = np.logspace(-3, 2, 6)
    rank = 3000
    for gam in gamma_range:
        K = exponential_kernel(X, X, gamma=gam)
        model = RFF(d=d, rank=rank, gamma=gam)
        model.fit()
        G = model.transform(X)
        L = G.dot(G.T)
        assert np.linalg.norm(L-K) < 10


def plot_exponential_nonstat():
    """ Test approximation of the exponential kernel. """
    np.random.seed(42)
    n = 100
    d = 1
    X = np.linspace(-10, 10, n).reshape((n, d))
    rank = 3000

    # Compare results using two different models
    model = RFF(d=d, rank=rank, gamma=0.01)
    model.fit()
    G = model.transform(X)
    L0 = G.dot(G.T)

    model = RFF_NS(d=d, rank=rank,
                   kwargs1={"gamma": 0.1},
                   kwargs2={"gamma": 0.01})
    model.fit()
    G = model.transform(X)
    L1 = G.dot(G.T)

    # Sample from GPs woth a given covariance structure
    F0 = mvn.rvs(mean=np.zeros((n,)), cov=L0, size=10000)
    F1 = mvn.rvs(mean=np.zeros((n,)), cov=L1, size=10000)
    for F, name in zip((F0, F1), ("stat", "nonstat")):
        f = F.mean(axis=0)
        s = F.std(axis=0)
        plt.plot(f, "-", label=name)
        plt.plot(f+s, "-", color="gray")
        plt.plot(f-s, "-", color="gray")
    plt.legend()
    plt.show()

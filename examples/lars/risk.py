import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import exponential_kernel, linear_kernel


def risk(X, f, sigma, N=100):
    """
    Estimate in-sample risk by bootstrap (Cp statistic).
    :param X: Design matrix.
    :param f: True function.
    :param sigma: Known noise variance.
    :param N: number of replications.
    :return:
    """
    # Sample N vectors
    n = X.shape[0]
    Y = mvn.rvs(mean=f.ravel(), cov=sigma * np.eye(n), size=N)

    # MSE
    Mu = Y * 0
    for i, y in enumerate(Y):
        Mu[i, :] = X.dot(np.linalg.lstsq(X, y)[0])

    # Covariances
    C = np.zeros(n)
    for i in range(n):
        C[i] = np.sum((Y[:, i] - Y[:, i].mean()) * Mu[:, i]) / (N - 1)

    mse = np.mean(np.power(Y - Mu, 2).sum(axis=1)) / sigma
    df = np.sum(C) / sigma
    c = mse - n + 2 * df
    return c, mse, df


def estimate_sigma(X, y):
    """ Estimate sigma from sample. """
    mu = X.dot(np.linalg.lstsq(X, y)[0]).ravel()
    return mu, np.var(y.ravel() - mu)


def estimate_risk(X, y, mu, sigma):
    """ Cheap risk estimate for certain linear models. """
    n, df = X.shape
    mse = np.linalg.norm(y.ravel() - mu.ravel())**2 / sigma
    return mse - n + 2 * df


# Unit tests
def test_estimate_sigma():
    """ Sigma estimates should be unbiased around the true value. """

    # Generate data
    sigma = 1.5
    N = 100
    m = 30
    n = 100
    X = np.linspace(-10, 10, n).reshape((n, 1))
    K = Kinterface(data=X, kernel=linear_kernel)[:, :]

    # Estimate sigma from a sample of data points
    est = np.zeros(N)
    for repl in range(N):
        y = mvn.rvs(mean=np.zeros(n), cov=K + sigma * np.eye(n))
        y = y - y.mean()
        inxs = np.random.choice(range(n), m, replace=False)
        _, est[repl] = estimate_sigma(K[inxs, :][:, inxs], y[inxs])

    assert np.mean(est) - np.std(est) < sigma < np.mean(est) + np.std(est)


# Experiments
def compare_risk():
    """ Evaluate risk on generated data. """
    n = 100
    sigma = 0.3
    X = np.linspace(-10, 10, n).reshape((n, 1))
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.5})[:, :]
    f = mvn.rvs(mean=np.zeros(n), cov=K)
    y = mvn.rvs(mean=f, cov=sigma * np.eye(n))

    # Estimate sigma from linear model
    # sigma_est = sigma
    mu_est, sigma_est = estimate_sigma(X, y)

    # Compute risk - bootstrap
    rank_range = range(2, n + 1)
    Cp = np.zeros(len(rank_range))
    mse = np.zeros(len(rank_range))
    df = np.zeros(len(rank_range))
    for i, r in enumerate(rank_range):
        inxs = np.linspace(0, n-1, r).astype(int)
        Cp[i], mse[i], df[i] = risk(K[:, inxs], f, sigma_est, N=100)

    # Compute risk - simple approximation
    Cp_est = np.zeros(len(rank_range))
    for i, r in enumerate(rank_range):
        inxs = np.linspace(0, n - 1, r).astype(int)
        mu = K[:, inxs].dot(np.linalg.lstsq(K[:, inxs], y)[0])
        Cp_est[i] = estimate_risk(K[:, inxs], y, mu, sigma_est)

    # Plot
    plt.figure()
    plt.plot(rank_range, Cp, label="boot")
    plt.plot(rank_range, Cp_est, label="simple")
    plt.xlabel("Rank")
    plt.ylabel("$C_p$")
    plt.legend()

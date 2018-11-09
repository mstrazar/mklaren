import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from mklaren.kernel.kernel import exponential_kernel
from examples.lars_vs_greedy.mklaren2 import find_bisector, least_sq
from sklearn.linear_model.ridge import Ridge

hlp = """
    Estimate in-sample risk.
    What is the expected value of the risk if the values of y are randomly shifted 
    (according to the noise variance).
"""


def Cp(G, f, sigma, N=100):
    """
    :param G: Design matrix.
    :param f: True function.
    :param sigma: Known variance.
    :param N: number of replications
    :return:
    """
    n = G.shape[0]
    Y = mvn.rvs(mean=f, cov=sigma * np.eye(n, n), size=N)
    Y = Y - Y.mean(axis=1).reshape((N, 1))

    # MSE
    Mu = Y * 0
    for i, y in enumerate(Y):
        Mu[i, :] = Ridge(alpha=0).fit(G, y).predict(G)

    # Covariances
    C = np.zeros(n)
    for i in range(n):
        C[i] = np.sum((Y[:, i] - Y[:, i].mean()) * (Mu[:, i])) / (N - 1)

    mse = np.mean(np.power(Y - Mu, 2).sum(axis=1)) / sigma
    df = np.sum(C) / sigma
    c = mse - n + 2 * df
    return c, df


# Generate design matrix
sigma = 0.4
n = 101
N = 1000
X = np.linspace(-10, 10, n).reshape(n, 1)
K = exponential_kernel(X, X, gamma=0.6)

# Generate a random smooth function
w = np.random.randn(n)
f = K.dot(w) - K.dot(w).mean()

# Estimate of the function for k basis functions
k_range = range(4, 30)
risk = np.zeros((len(k_range),))
df = np.zeros((len(k_range),))

for i, k in enumerate(k_range):
    inxs = np.linspace(0, len(X)-1, k).astype(int)
    G = K[:, inxs]
    risk[i], df[i] = Cp(G=G, f=f, sigma=sigma, N=N)

# Risk vs. num of basis
kmin = k_range[np.argmin(risk)]
plt.close("all")
plt.figure()
plt.plot([kmin, kmin], [0, 0], "v", markersize=3)
plt.plot(k_range, risk)
plt.xlabel("Num. basis functions (k)")
plt.ylabel("Risk ($C_p$ statistic)")


# Plot best fit
kt = kmin
inxs = np.linspace(0, len(X)-1, kt).astype(int)
G = K[:, inxs]
y = f + sigma * np.random.randn(n)
mu = Ridge(alpha=0).fit(G, y).predict(G)
plt.figure()
plt.plot(y, ".", label="samples")
plt.plot(f, "b-", label="f")
plt.plot(mu, label="$\mu$ (k=%d)" % kt)
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")


plt.figure()
plt.plot(y, y-mu, ".")
plt.xlabel("y")
plt.ylabel("y-mu")
plt.title("Residual plot")

"""
Comparison of selction of pivots in one dimension for MKL & CSI (later SPGP).

"""
# import matplotlib
# matplotlib.use("Agg")

import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt

from mklaren.kernel.kernel import exponential_kernel, linear_kernel, poly_kernel
from mklaren.kernel.kinterface import Kinterface
from scipy.stats import multivariate_normal as mvn

from examples.features import Features
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from sklearn.metrics import mean_squared_error as mse


# Complete data range
p = 10
n = 5
sigma2 = 1
Xt = np.linspace(-2, 2, 2 * n).reshape((2 * n, 1))

feats = Features(degree=6)
Ft = feats.fit_transform(Xt)

# True kernel and signal
K = Kinterface(data=Ft, kernel=linear_kernel)
f = mvn.rvs(mean=np.zeros((2 * n,)), cov=K[:, :] + sigma2 * np.eye(2*n, 2*n), size=1).ravel()

# Plot function
# plt.figure()
# plt.plot(Xt.ravel(), f, ".")
# plt.show()


# Eigenvalue of the exponential kernel
plt.figure()
for gi, g in enumerate(np.logspace(-1, 3, 4)):
    K = Kinterface(data=Xt, kernel=exponential_kernel, kernel_args={"gamma": g})
    vals, vecs = np.linalg.eig(K[:, :])
    plt.plot(vals, label="$\gamma=%0.2f,\  \sigma^2=%0.4f$" % (g, 1.0/g), linewidth=2)
plt.legend()
plt.show()


# Random kernels converge to indentity (pure-noise) covariance
n = 5
plt.figure()
for pi, p in enumerate(np.logspace(1, 6, 6)):
    Y = np.random.randn(n, p)
    K = Kinterface(data=Y, kernel=linear_kernel, row_normalize=True)
    vals, vecs = np.linalg.eig(K[:, :])
    vals = sorted(vals, reverse=True)
    plt.plot(vals, label="p=%d" % p, linewidth=2)

    print("p=%d"% p)
    print(K[:, :])
    print()
plt.legend()
plt.show()
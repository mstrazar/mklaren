# Low rank kernel function vs. sampling from a GP
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mklaren.kernel.kernel import exponential_kernel, periodic_kernel
from mklaren.kernel.kinterface import Kinterface
from scipy.stats import multivariate_normal as mvn
from mklaren.regression.ridge import RidgeLowRank

outdir = "/Users/martin/Dev/mklaren/examples/output/function_space/"
n = 100
N = 1000
X = np.linspace(-10, 10, n).reshape((n, 1))
y = 2 * X + 0.5
gamma_range = np.logspace(-3, 3, 7)

# Random inducing indices
inxs = [10, 30, 70]
rank = len(inxs)

# Multiple lengthscales kernels
Ks = [Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": g})
      for g in gamma_range]

# Kernel matrix ad-hoc approximation with fixed index set
inxs = [10, 20, 50, 90]
rank = len(inxs)
Gs = [K[:, inxs].dot(sp.linalg.sqrtm(np.linalg.inv(K[inxs, inxs]))) for K in Ks]

# Sum features over all kernels
Q = sum(Gs)

# Create an artifical weight vector only allowing short lengthscales
# Centered on the lengthscale for on of the inducing points
w = np.zeros((Gs[0].shape[1], 1))
w[:rank] = 1
yp = Gs[0].dot(w)

# Targets
noise = 0.00
yn = noise * np.random.randn(n, 1).reshape((n, 1))

# Solvable within a combined feature space
# This computation should be exact, but the problem has a bad condition
# number. The coefficients are HUGE and its impossible to retrieve
# the true solution.

lbd = 0.0
# K_app = sum([G.dot(G.T) for g in Gs])   # exact
# K_app = sum([K[:, :] for K in Ks])            # exact
K_app = Q.dot(Q.T)            # inexact
rank = np.linalg.matrix_rank(K_app)
a_app = sp.linalg.solve(K_app + lbd * np.eye(n, n), yp+yn)     # Modeling
y_app = K_app.dot(a_app)
plt.figure()
plt.ylim(yp.min(), yp.max())
plt.plot(X.ravel(), yp, label="True")
plt.plot(X.ravel(), y_app, label="App")
plt.legend(title="Rank=%d" % rank)
plt.show()

# Estimate difference
d = np.linalg.norm(yp - y_app)
a = np.linalg.norm(a_app)
a1 = np.linalg.norm(a_app, ord=1)
print("Norm of difference: %f" % d)
print("Norm1 of alpha: %f" % a1)
print("Norm2 of alpha: %f" % a)

# Retrieving the true w
# w_app = G.T.dot(a_app)
# dw = np.linalg.norm(w - w_app)
# print("Norm of weight difference: %f" % dw)

# Transform test points outside of domain
Xp = np.linspace(10, 12, 10).reshape((10, 1))
Xt = model.transform([Xp]*len(gamma_range))
xp  = Xt.dot(w)
xp_app = Xt.dot(w_app)
plt.figure()
plt.plot(Xp.ravel(), xp, label="True")
plt.plot(Xp.ravel(), xp_app, label="App")
plt.legend()
plt.show()



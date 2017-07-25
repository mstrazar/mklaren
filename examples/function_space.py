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

# Kernel matrix
Ks = [Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": g})
      for g in gamma_range]

# Only to construct low-rank features
model = RidgeLowRank(rank=2, method="nystrom")
model.fit(Ks, y)

# Create an artifical weight vector only allowing showrt lengthscales
G = np.hstack(model.Gs)
w = np.zeros((G.shape[1], 1))
w[:2] = 1
yp = G.dot(w)

# Solvable within a combined feature space
lbd = 1e-5
K_app = sum([g.dot(g.T) for g in model.Gs])
rank = np.linalg.matrix_rank(K_app)
a_app = sp.linalg.solve(K_app + lbd * np.eye(n, n), yp)     # Modeling
y_app = K_app.dot(a_app)
plt.figure()
plt.ylim(yp.min(), yp.max())
plt.plot(yp, label="True")
plt.plot(y_app, label="App")
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
w_app = G.T.dot(a_app)
dw = np.linalg.norm(w - w_app)
print("Norm of weight difference: %f" % dw)
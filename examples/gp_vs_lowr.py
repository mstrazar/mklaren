# Low rank kernel function vs. sampling from a GP
import os
import numpy as np
import matplotlib.pyplot as plt
from mklaren.kernel.kernel import exponential_kernel, periodic_kernel
from mklaren.kernel.kinterface import Kinterface
from scipy.stats import multivariate_normal as mvn

outdir = "/Users/martin/Dev/mklaren/examples/output/gp_vs_lowr/"
n = 100
N = 1000
X = np.linspace(-10, 10, n).reshape((n, 1))

# Random inducing indices
inxs = [10, 30, 70]
rank = len(inxs)

# Kernel matrix
K = Kinterface(data=X, kernel=periodic_kernel, kernel_args={"l": 1.0})[:, :] \
    + Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.1})[:, :]

Ki = np.linalg.inv(K[inxs, :][:, inxs])
Ka = K[:, inxs]
K_app = Ka.dot(Ki).dot(Ka.T)

# GP-samples from low-rank matrix
samples = mvn.rvs(mean=np.zeros((n,)), cov=K_app, size=N)

# Random matrix with weights
alphas = np.zeros((N, n))
alphas[:, inxs] = mvn.rvs(mean=np.zeros((rank,)), cov=np.eye(rank, rank), size=N)
funcs = alphas.dot(K_app)


# Compare equivalent ways to generate the sample functions
fname = os.path.join(outdir, "samples_GP_sum.pdf")
plt.figure()
plt.title("GP-samples from low-rank kernel matrix")
plt.plot(samples.T)
plt.xlabel("Input space")
plt.ylabel("y")
plt.savefig(fname)
plt.close()

fname = os.path.join(outdir, "samples_RKHS_sum.pdf")
plt.figure()
plt.title("Random functions in approximated RKHS")
plt.plot(funcs.T)
plt.xlabel("Input space")
plt.ylabel("y")
plt.savefig(fname)
plt.close()
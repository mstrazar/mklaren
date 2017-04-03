"""
Comparison of selction of pivots in one dimension for MKL & CSI (later SPGP).

"""
import matplotlib
matplotlib.use("Agg")

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from scipy.stats import multivariate_normal as mvn

from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank


# Create output directory
d = datetime.datetime.now()
dname = os.path.join("output", "one_dim_pivots", "%d-%d-%d" % (d.year, d.month, d.day))
subdname = os.path.join(dname, "%02d" % len(os.listdir(dname)))
if not os.path.exists(subdname):
    os.makedirs(subdname)


# Experiment parameters
n = 100
s = 5           # Signal range
nsig = 10
noise = 0.01
rank = 2
delta = 10
repeats = 10
lbd = 0.01
methods = ["Mklaren", "ICD", "Nystrom", "CSI"]

X = np.linspace(-1, 1, n).reshape((n, 1))
Xt = np.linspace(-2, 2, n).reshape((n, 1))

# True kernel and signal
K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 10})

for repl in range(repeats):


    # Random variations in the signal
    f = np.zeros((n,))
    pi = np.random.choice(range(X.shape[0]))  # Choose a random point on x
    f[pi - s:pi + s] = mvn.rvs(mean=np.zeros((2 * s,)),
                               cov=K[pi - s:pi + s, pi - s:pi + s], size=1).ravel()

    # Pure signal
    f = mvn.rvs(mean=np.zeros((n,)), cov=K[:, :], size=1).ravel()
    y = mvn.rvs(mean=f, cov=np.eye(n, n) * noise, size=1).ravel()

    for method in methods:


        if method == "Mklaren":# Mklaren model
            model = Mklaren(rank=rank, delta=delta, lbd=lbd)
            model.fit([K], y)
            yp = model.predict([Xt])
            u = X[model.data[0]["act"], :].ravel()

        elif method == "ICD":
            model = RidgeLowRank(rank=rank,
                                 method="icd",
                                 lbd=lbd)
            model.fit([K], y)
            yp = model.predict([Xt])
            u = X[model.As[0], :].ravel()

        elif method == "CSI":
            model = RidgeLowRank(rank=rank,
                                 method="csi",
                                 method_init_args={"delta": delta},
                                 lbd=lbd)
            model.fit([K], y)
            yp = model.predict([Xt])
            u = X[model.As[0], :].ravel()

        elif method == "Nystrom":
            model = RidgeLowRank(rank=rank,
                                 method="nystrom",
                                 lbd=lbd,
                                 method_init_args={"lbd": lbd})
            model.fit([K], y)
            yp = model.predict([Xt])
            u = X[model.As[0], :].ravel()


        fname = os.path.join(subdname, "%02d_%s.pdf" % (repl, method))
        print("Writing to %s ..." % fname)

        # Plot results
        plt.figure()
        plt.title(method)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.plot(X.ravel(), f, "-", color="gray", label = "f(x)")
        plt.plot(X.ravel(), y, ".", color="gray", markersize=10.0, label="y(f)")
        plt.plot(Xt.ravel(), yp, "-", color="red", linewidth=2, label="Model")
        plt.plot(u, [-2]*len(u), "+", color="red", markersize=10, label="active set")
        plt.xlim(Xt[0], Xt[-1])
        plt.ylim(-3, 3)
        plt.legend()
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
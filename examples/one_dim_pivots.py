"""
Comparison of selction of pivots in one dimension for MKL & CSI (later SPGP).

"""
import matplotlib
matplotlib.use("Agg")

import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt

from mklaren.kernel.kernel import exponential_kernel, linear_kernel
from mklaren.kernel.kinterface import Kinterface
from scipy.stats import multivariate_normal as mvn

from features import Features
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from sklearn.metrics import mean_squared_error as mse


# Create output directory
d = datetime.datetime.now()
dname = os.path.join("output", "one_dim_pivots", "%d-%d-%d" % (d.year, d.month, d.day))
count = len(os.listdir(dname))
subdname = os.path.join(dname, "%02d" % count)
if not os.path.exists(subdname):
    os.makedirs(subdname)

# Initialize csv
header = ["repl", "method", "n", "rank", "lbd", "mse_te", "mse_va"]
csvname = os.path.join(subdname, "_results.csv")
writer = csv.DictWriter(open(csvname, "w", buffering=0), fieldnames=header)
writer.writeheader()
print("Writing to %s ... " % csvname)


# Experiment parameters
n = 300
s = 5           # Signal range
nsig = 10
noise = 0.01
rank = 5
delta = 5
repeats = 20
lbd_range = [0] + list(np.logspace(-2, 2, 5))
methods = ["Mklaren", "ICD", "Nystrom", "CSI"]


# Plot results
def plot_fit(fname, X, f, y, yp, tr, u, method):
    """ Plot the results on a figure """
    plt.figure()
    plt.title(method)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(X.ravel(), f, "-", color="gray", label = "f(x)")
    plt.plot(X[tr].ravel(), y, ".", color="gray", markersize=10.0, label="y(f)")
    plt.plot(X, yp, "-", color="red",  linewidth=2,    label="Model")
    plt.plot(u, [-2]*len(u), "+", color="red",  markersize=10,  label="active set")
    plt.xlim(X[0], X[-1])
    plt.legend()
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


# Complete data range
Xt = np.linspace(-2, 2, 2 * n).reshape((2 * n, 1))

feats = Features(degree=6)
Ft = feats.fit_transform(Xt)

# True kernel and signal
K = Kinterface(data=Ft, kernel=linear_kernel)


# Smartsplit of test set
tr = range(n-n/4, n+n/4)
va = range(n-n/2, n-n/4) + range(n+n/4, n + n/2)

# tr = range(n-n/2, n+n/2, 2)                 # training set
# va = range(n-n/2+1, n+n/2, 2)               # validation set
te = range(0, n/2) + range(n + n/2, 2 * n)  # test (extrapolation) set

# Corresponding covariance matrices
K    = Kinterface(data=Ft, kernel=linear_kernel)
K_te = Kinterface(data=Ft[te], kernel=linear_kernel)
K_tr = Kinterface(data=Ft[tr], kernel=linear_kernel)
K_va = Kinterface(data=Ft[va], kernel=linear_kernel)


for repl in range(repeats):

    # Pure signal and range of data is around zero
    f = mvn.rvs(mean=np.zeros((2 * n,)), cov=K[:, :], size=1).ravel()
    y = mvn.rvs(mean=f, cov=np.eye(2 * n, 2 * n) * noise, size=1).ravel()

    rows = list()
    for method in methods:

        # Evaluate function at training points
        lbd_best = 0
        mse_best = float("inf")
        u_best = yp_te = yp_all = None
        mse_final = 0

        # Select lambda with cross-validation
        for lbd in lbd_range:
            if method == "Mklaren":# Mklaren model
                model = Mklaren(rank=rank, delta=delta, lbd=lbd)
                model.fit([K_tr], y[tr])
                u = Xt[tr, :][model.data[0]["act"], :].ravel()

            elif method == "ICD":
                model = RidgeLowRank(rank=rank,
                                     method="icd",
                                     lbd=lbd)
                model.fit([K_tr], y[tr])
                u = Xt[tr][model.As[0], :].ravel()

            elif method == "CSI":
                model = RidgeLowRank(rank=rank,
                                     method="csi",
                                     method_init_args={"delta": delta},
                                     lbd=lbd)
                model.fit([K_tr], y[tr])
                u = Xt[tr][model.As[0], :].ravel()

            elif method == "Nystrom":
                model = RidgeLowRank(rank=rank,
                                     method="nystrom",
                                     lbd=lbd,
                                     method_init_args={"lbd": lbd})
                model.fit([K_tr], y[tr])
                u = Xt[tr][model.As[0], :].ravel()

            # Determine best scores and parameters
            yp_va = model.predict([Ft[va]])
            mse_va = mse(yp_va, y[va])
            if mse_va < mse_best:
                yp_all = model.predict([Ft])
                yp_te = model.predict([Ft[te]])
                mse_final = mse(yp_te, y[te])
                mse_best = mse_va
                u_best = u
                lbd_best = lbd

        print("Optimal lbd: %f" % lbd_best)
        row = {"repl": repl, "rank": rank, "lbd": lbd_best,
               "mse_te": mse_final, "n": n,
               "mse_va": mse_best,
               "method": method,}
        rows.append(row)

        fname = os.path.join(subdname, "%02d_%s.pdf" % (repl, method))
        print("Writing to %s ..." % fname)
        plot_fit(fname, Xt, f, y[tr], yp_all, tr, u_best, method)

    # Write results to file
    if len(rows) == len(methods):
        writer.writerows(rows)

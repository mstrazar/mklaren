"""
Motivation: If one is allowed to sample columns with regularization,
this leads to lower errors at high noise levels.

Comparison of the three pivot selection methods with varying noise rates
on a simple Gaussian Process signal.

There is no training and test set, just comparison with recovering true
signal.

How to select lambda?
    - Display results for all lambda.

Add more kernels?

"""
import matplotlib
matplotlib.use("Agg")

import csv
import datetime
import os
import itertools as it

import numpy as np
from scipy.stats import multivariate_normal as mvn, pearsonr
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt



# Create output directory
d = datetime.datetime.now()
dname = os.path.join("output", "snr", "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname):
    os.makedirs(dname)
fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
print("Writing to %s ..." % fname)

header = ["repl", "method", "n", "gamma", "lbd", "snr", "rank", "noise", "mse_sig", "mse_rel", "pr_rho", "pr_pval"]
writer = csv.DictWriter(open(fname, "w", buffering=0),
                        fieldnames=header, quoting=csv.QUOTE_ALL)
writer.writeheader()

# Parameters
gamma_range = 2**(np.linspace(-2, 2, 5))  # Arbitrary kernel hyperparameters
delta = 10                                # Arbitrary look-ahead parameter

# Objective experimentation;
n_range = [100, 300, 500]                 # Vaste enough range of dataset sizes (which are full-rank)
noise_range = np.logspace(-3, 3, 7)       # Range of noise levels
repeats = range(10)                       # Number of repeats
rank_percents = [0.05, 0.1, 0.15]         # Rank percentages given n
methods = ["Mklaren", "CSI", "ICD"]
lbd_range = [0, 0.1, 0.3, 1, 3, 10, 30, 100]  # Vast enough range, such that methods should be able to capture optimum somewhere

count = 0
for repl, gamma, n, noise, lbd, rp in it.product(repeats, gamma_range, n_range,
                                                 noise_range, lbd_range, rank_percents):
    rank = max(5, int(rp * n))

    # Generate data
    X = np.linspace(-10, 10, n).reshape((n, 1))
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": gamma})

    f = mvn.rvs(mean=np.zeros((n, )), cov=K[:, :])
    y = mvn.rvs(mean=f, cov=noise * np.eye(n, n))

    # Mklaren
    yp = None
    rows = []
    for method in methods:
        # The difference is only in including lambda at the time of pivot selection
        try:
            if method == "Mklaren":
                mkl = Mklaren(rank=rank,
                              delta=delta, lbd=lbd)
                mkl.fit([K], y)
                yp = mkl.predict([X])

            elif method == "ICD":
                icd = RidgeLowRank(rank=rank, lbd=lbd,
                                   method="icd")
                icd.fit([K], y)
                yp = icd.predict([X])

            elif method == "CSI":
                icd = RidgeLowRank(rank=rank, lbd=lbd,
                                   method="csi", method_init_args={"delta": delta})
                icd.fit([K], y)
                yp = icd.predict([X])
        except Exception as e:
            print("%s exception: %s" % (method,  e.message))
            continue

        # Metrics
        mse_sig = mse(yp, f)
        mse_rel = mse(yp, f) / np.var(y)
        snr = np.var(f) / noise
        pr_rho, pr_pval = pearsonr(yp, f)

        row = {"repl": repl, "method": method, "n": n, "snr": snr, "lbd": lbd,
               "rank": rank, "noise": np.round(np.log10(noise), 2), "mse_sig": mse_sig,
               "mse_rel": mse_rel, "pr_rho": pr_rho, "pr_pval": pr_pval, "gamma": gamma}
        rows.append(row)

    if len(rows) == len(methods):
        writer.writerows(rows)
        count += len(rows)
        print("%s Written %d rows (n=%d)" % (str(datetime.datetime.now()), count, n))


if False:
    # Generate data
    n = 300
    lbd = 1
    noise = 1
    X = np.linspace(-10, 10, n).reshape((n, 1))
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 1})

    f = mvn.rvs(mean=np.zeros((n,)), cov=K[:, :])
    y = mvn.rvs(mean=f, cov=noise * np.eye(n, n))

    # Fit MKL
    mkl = Mklaren(rank=rank,
                   delta=delta, lbd=lbd)
    mkl.fit([K], y)
    yp = mkl.predict([X])

    # Fit ICD
    icd = RidgeLowRank(rank=rank, lbd=lbd, method="icd")
    icd.fit([K], y)
    yp_icd = icd.predict([X])

    # Fit CSI
    csi = RidgeLowRank(rank=rank, lbd=lbd, method="csi", method_init_args={"delta": delta})
    # csi.fit([K], y)
    # yp_csi = csi.predict([X])


    # Plot archive snippet
    # Plot data
    plt.figure()
    plt.plot(X.ravel(), f, "--", color="gray")
    plt.plot(X.ravel(), y, ".", color="black")
    plt.plot(X.ravel(), yp, "-", color="blue", linewidth=2)
    plt.plot(X.ravel(), yp_icd, "-", color="green", linewidth=2)
    # plt.plot(X.ravel(), yp_csi, "-", color="red", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    print("MKL", pearsonr(yp, f))
    # print("CSI", pearsonr(yp_csi, f))
    print("ICD", pearsonr(yp_icd, f))

    print("MKL", mse(yp, f))
    print("ICD", mse(yp_icd, f))

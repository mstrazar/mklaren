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
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
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
noise_range = np.logspace(-2, 2, 7)       # Range of noise levels
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


def test(n=100, noise=0.1, rank=10, lbd=0.1, seed=None):
    """
    Sample data from a Gaussian process and compare fits with the sum of kernels
    versus list of kernels.
    :param n: Number of data points.
    :param noise:  Noise variance.
    :param rank:  Approximation rank.
    :param lbd: Regularization parameter.
    :param seed: Random seed.
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    # small gamma, large lengthscale -> small gamma, small frequency
    gamma_range = np.logspace(-1, 3, 5)

    # Generate data
    X = np.linspace(-10, 10, n).reshape((n, 1))

    # Kernel sum
    Ksum = Kinterface(data=X, kernel=kernel_sum,
                   kernel_args={
                       "kernels": [exponential_kernel] * len(gamma_range),
                       "kernels_args": [{"gamma": g} for g in gamma_range]})

    # Sum of kernels
    Klist = [Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": g})
            for g in gamma_range]

    f = mvn.rvs(mean=np.zeros((n,)), cov=Ksum[:, :])
    y = mvn.rvs(mean=f, cov=noise * np.eye(n, n))

    # Fit MKL for kernel sum and
    mkl = Mklaren(rank=rank,
                  delta=10, lbd=lbd)
    mkl.fit(Klist, y)
    yp_Klist = mkl.predict([X] * len(Klist))
    Axs = [X[mkl.data.get(gi, {}).get("act", [])].ravel() for gi in range(len(gamma_range))]

    mkl.fit([Ksum], y)
    yp_Ksum = mkl.predict([X])

    # Frequency scales
    ymin = int(np.absolute(np.min(y)))
    Gxs = [np.linspace(-10, 10, 5 + 10 * g) for g in gamma_range]
    Gys = range(-ymin-len(Gxs), -ymin)

    # Correlation
    rho_Klist, _ = pearsonr(yp_Klist, f)
    rho_Ksum, _ = pearsonr(yp_Ksum, f)

    # Plot a summary figure
    plt.figure()

    # Plot signal
    x = X.ravel()
    plt.plot(x, y, "k.")
    plt.plot(x, f, "r--")
    plt.plot(x, yp_Klist, "g-", label="Klist ($\\rho$=%.2f)" % rho_Klist)
    plt.plot(x, yp_Ksum, "b-", label="Ksum ($\\rho$=%.2f)" % rho_Ksum)

    # Plot freqency scales
    for gi, (gx, gy) in enumerate(zip(Gxs, Gys)):
        plt.plot(gx, [gy] * len(gx), "|", color="gray")
        if len(Axs[gi]):
            print("Number of pivots at gamma  %d: %d" % (gi, len(Axs[gi])))
            plt.plot(Axs[gi], [gy]*len(Axs[gi]), "x", color="green", markersize=6)
    plt.title("n=%d, noise=%.3f, rank=%d, lambda=%0.3f" % (n, noise, rank, lbd))
    plt.legend()
    plt.xlim((-11, 11))
    plt.show()
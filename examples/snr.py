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

def process():

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


def test(n=100, noise=0.1, rank=10, lbd=0.1, seed=None,
         P=5, gmin=-1, gmax=4, delta=10, plot=True, inducing_mode="uniform"):
    """
    Sample data from a Gaussian process and compare fits with the sum of kernels
    versus list of kernels.
    :param n: Number of data points.
    :param noise:  Noise variance.
    :param rank:  Approximation rank.
    :param lbd: Regularization parameter.
    :param seed: Random seed.
    :param inducing_mode: How to choose inducing points.
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    # small gamma, large lengthscale -> small gamma, small frequency
    gamma_range = np.logspace(gmin, gmax, P)

    # Generate data
    X = np.linspace(-10, 10, n).reshape((n, 1))
    Xp = np.linspace(-10, 10, 500).reshape((500, 1))

    # Kernel sum
    Ksum = Kinterface(data=X, kernel=kernel_sum,
                   kernel_args={
                       "kernels": [exponential_kernel] * len(gamma_range),
                       "kernels_args": [{"gamma": g} for g in gamma_range]})

    # Sum of kernels
    Klist = [Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": g})
            for g in gamma_range]

    a = np.arange(n, dtype=float)
    if inducing_mode == "uniform":
        p = None
    elif inducing_mode == "biased":
        p = (a ** 2 / (a ** 2).sum())
    else:
        raise ValueError(inducing_mode)
    inxs = np.random.choice(a, p=p, size = rank, replace = False)
    Kny = Ksum[:, inxs].dot(np.linalg.inv(Ksum[inxs, inxs])).dot(Ksum[inxs, :])
    f = mvn.rvs(mean=np.zeros((n,)), cov=Kny)

    # f = mvn.rvs(mean=np.zeros((n,)), cov=Ksum[:, :])
    y = mvn.rvs(mean=f, cov=noise * np.eye(n, n))

    # Fit MKL for kernel sum and
    mkl = Mklaren(rank=rank,
                  delta=delta, lbd=lbd)
    mkl.fit(Klist, y)
    y_Klist = mkl.predict([X] * len(Klist))
    yp_Klist = mkl.predict([Xp] * len(Klist))
    active_Klist = [mkl.data.get(gi, {}).get("act", [])for gi in range(P)]
    anchors_Klist = [X[mkl.data.get(gi, {}).get("act", [])].ravel() for gi in range(P)]


    mkl.fit([Ksum], y)
    y_Ksum = mkl.predict([X])
    yp_Ksum = mkl.predict([Xp])

    # Fit CSI
    csi = RidgeLowRank(rank=rank, lbd=lbd,
                       method="csi", method_init_args={"delta": delta},)
    csi.fit([Ksum], y)
    y_csi = csi.predict([X])
    yp_csi = csi.predict([Xp])
    active_csi = [csi.active_set_[gi] for gi in range(P)]
    anchors_csi = [X[csi.active_set_[gi]] for gi in range(P)]

    # Frequency scales
    ymin = int(np.absolute(np.min(y)))
    Gxs = [np.linspace(-10, 10, 5 + 10 * g) for g in gamma_range]
    Gys = range(-ymin-len(Gxs), -ymin)

    # Correlation
    rho_Klist, _ = pearsonr(y_Klist, f)
    rho_Ksum, _ = pearsonr(y_Ksum, f)
    rho_csi, _ = pearsonr(y_csi, f)

    # Plot a summary figure
    if plot:
        plt.figure()

        # Plot signal
        x = X.ravel()
        xp = Xp.ravel()
        plt.plot(x, y, "k.")
        plt.plot(x, f, "r--")
        plt.plot(xp, yp_Klist, "g-", label="Klist ($\\rho$=%.2f)" % rho_Klist)
        # plt.plot(xp, yp_Ksum, "b-", label="Ksum ($\\rho$=%.2f)" % rho_Ksum)
        plt.plot(xp, yp_csi, "r-", label="CSI ($\\rho$=%.2f)" % rho_csi)

        # Plot freqency scales
        for gi, (gx, gy) in enumerate(zip(Gxs, Gys)):
            plt.plot(gx, [gy] * len(gx), "|", color="gray")
            if len(anchors_Klist[gi]):
                print("Number of pivots at gamma  %d: %d" % (gi, len(anchors_Klist[gi])))
                plt.plot(anchors_Klist[gi], [gy] * len(anchors_Klist[gi]), "^", color="green", markersize=8, alpha=0.6)
            if len(anchors_csi[gi]):
                plt.plot(anchors_csi[gi], [gy] * len(anchors_Klist[gi]), "^", color="red", markersize=8, alpha=0.6)
        plt.title("n=%d, noise=%.3f, rank=%d, lambda=%0.3f" % (n, np.max(noise), rank, lbd))
        ylim = plt.gca().get_ylim()
        plt.legend()
        plt.xlim((-11, 11))
        plt.ylim((ylim[0]-1, ylim[1]))
        plt.show()
    else:
        return inxs, rho_Klist, rho_csi, active_Klist, active_csi


if __name__ == "__main__":
    # process()
    n = 100
    noise = 1
    rank = 3
    lbd = 0
    repeats = 500

    out_dir = "examples/output/snr/images/"

    noise_models = ("fixed", "increasing")
    sampling_models = ("uniform", "biased")

    for noise_model, inducing_mode in it.product(noise_models, sampling_models):
        if noise_model == "fixed":
            noise = 1
        else:
            noise = np.logspace(-2, 2, n)

        avg_anchors = dict()

        for seed in range(repeats):
            r = test(n=n, P=1, noise=noise, gmin=0, gmax=1, rank=rank, seed=seed, lbd=lbd,
                     plot=False, inducing_mode=inducing_mode)
            inxs, rho_Klist, rho_csi, active_Klist, active_csi = r
            avg_anchors["Mklaren"] = avg_anchors.get("Mklaren", []) + active_Klist[0]
            avg_anchors["CSI"] = avg_anchors.get("CSI", []) + active_csi[0]
            avg_anchors["True"] = avg_anchors.get("True", []) + list(inxs)

        # Compare distributions
        bins =  np.array([  4. ,  13.5,  23. ,  32.5,  42. ,  51.5,  61. ,  70.5,  80. , 89.5,  99. ])
        avg_mkl, _ = np.histogram(avg_anchors["Mklaren"], normed=True, bins=bins)
        avg_csi, _ = np.histogram(avg_anchors["CSI"], normed=True, bins=bins)
        avg_tru, _ = np.histogram(avg_anchors["True"], normed=True, bins=bins)
        mkl_tot_var = np.sum(np.absolute(avg_mkl - avg_tru))
        csi_tot_var = np.sum(np.absolute(avg_csi - avg_tru))

        # Plot histograms
        fname = os.path.join(out_dir, "noisy_sampling_%s-%s.pdf" % (noise_model, inducing_mode))
        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].hist(avg_anchors["True"], color="gray", label="True")
        ax[1].hist(avg_anchors["Mklaren"], color="green", label="Mklaren (%.3f)" % mkl_tot_var)
        ax[2].hist(avg_anchors["CSI"], color="blue", label="CSI (%.3f)" % csi_tot_var)
        for i in range(len(ax)):
            ax[i].legend()
            ax[i].set_xlim((0, n))
        plt.savefig(fname)
        plt.close()
        print("Written %s" % fname)
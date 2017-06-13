from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank

from examples.snr.snr import generate_data as snr_data, test as snr_test, plot_signal

from time import time
from datetime import datetime
from itertools import product
import numpy as np
import csv
import os
import itertools as it

import matplotlib.pyplot as plt


def generate_data(n, max_rank, p_tr, gamma_range=(1,), max_scales=1):
    """
    Generate data with a given number of inducing points within the training set.
    Compute signalusing the Nystrom approximation.
    :param n:
        Number of data points.
    :param max_rank:
        Number of inducing points.
    :param max_scales:
        Number of inducing lengthscales.
    :param gamma_range:
        Range of lengthscales.
    :param p_tr:
        Fraction of training.
    :return:
    """
    # Rank is known because the data is simulated
    X = np.random.rand(n, max_rank)
    X = (X - X.mean(axis=0)) / np.std(X, axis=0)

    # Fit one kernel
    tr_inxs = range(int(p_tr * n))
    te_inxs = range(int(p_tr * n), n)
    X_tr = X[tr_inxs, :]
    X_te = X[te_inxs, :]

    # One kernel where funciton is the sum
    K_tr = Kinterface(data=X_tr,
                    kernel=kernel_sum,
                    kernel_args={"kernels": [exponential_kernel] * len(gamma_range),
                                 "kernels_args": [{"gamma": g} for g in gamma_range]})

    # Same thing, but using a list of kernels instead
    Ks_all = [Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": g}) for g in gamma_range]
    Ks_tr = [Kinterface(data=X_tr, kernel=exponential_kernel, kernel_args={"gamma": g}) for g in gamma_range]

    # Signal is defined using a random subset of the training set
    siginxs = np.random.choice(tr_inxs, size=max_rank, replace=False)
    alpha = np.zeros((n, 1))
    alpha[siginxs] = np.random.rand(len(siginxs), 1)

    # Signal is defined in the span of the training set
    # Use Nystrom approximation to compute signal efficiently

    # Only a few lengthscales are relevant
    sigscales = np.random.choice(range(len(gamma_range)), replace=False, size=max_scales)
    K_s  = sum((Ks_all[i][siginxs, :] for i in sigscales))
    K_ss = sum((Ks_all[i][siginxs, siginxs] for i in sigscales))

    Ka = K_s.dot(alpha)
    KiKa = np.linalg.inv(K_ss).dot(Ka)
    y_true = K_s.T.dot(KiKa)
    y_true = y_true - y_true.mean()
    y_tr = y_true[tr_inxs]
    y_te = y_true[te_inxs]

    return Ks_tr, K_tr, X_tr, X_te, y_tr, y_te




def process2():
    repeats = 10
    n = 1000
    inducing_mode = "biased"
    noise = 0.1
    gamma = 10
    max_rank = 30
    methods = ("Mklaren", "ICD", "Nystrom")
    lbd_range   = np.power(10, np.linspace(-2, 2, 5))           # Arbitrary lambda hyperparameters
    rank_range  = range(2, max_rank, int(0.1 * max_rank))

    # Create output directory
    d = datetime.now()
    dname = os.path.join("output", "timing", "%d-%d-%d" % (d.year, d.month, d.day))
    if not os.path.exists(dname):
        os.makedirs(dname)
    fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
    print("Writing to %s ..." % fname)

    # Output file
    header = ["repl", "method", "n", "max_rank", "rank", "lbd", "time", "expl_var"]
    writer = csv.DictWriter(open(fname, "w", buffering=0),
                            fieldnames=header, quoting=csv.QUOTE_ALL)
    writer.writeheader()


    count = 0
    for seed in range(repeats):

        Ksum, Klist, inxs, X, Xp, y, f = snr_data(n=n, inducing_mode=inducing_mode, gamma_range=[gamma],
                                                  rank=max_rank, seed=seed, noise=noise)

        for rank, lbd in it.product(rank_range, lbd_range):
            jx = inxs[:rank]
            r = snr_test(Ksum, Klist, jx, X, Xp, y, f, methods=methods, lbd=lbd)

            rows = list()
            for method in methods:
                row = {"n": n, "method": method, "rank": rank,
                       "time": r[method]["time"], "expl_var": r[method]["evar"],
                       "repl": seed, "lbd": lbd, "max_rank": max_rank}
                rows.append(row)

            count += len(rows)
            writer.writerows(rows)
            print("Written %d rows" % count)


def process():
    # Fixed hyper parameters
    repeats = 10                                                # Number of replicas.
    p_tr = 0.6                                                  # Fraction of test set
    # range_n = map(int, 1.0/p_tr * np.array([1e5, 3e5, 1e6]))  # Number of samples in TRAINING set.
    range_n = map(int, 1.0 / p_tr * np.array([1e3, 3e3, 1e4]))  # Number of samples in TRAINING set.
    methods = ["Mklaren", "ICD", "Nystrom"]                     # Methods
    delta = 10                                                  # Lookahead columns
    max_rank = 30                                               # Max. rank and number of indicuing points.
    max_scales = 2                                              # Two relevant lengthscales
    gamma_range = np.power(10, np.linspace(-3, -1, 5))          # Arbitrary kernel hyperparameters
    lbd_range   = np.power(10, np.linspace(-2, 2, 5))           # Arbitrary lambda hyperparameters

    # Create output directory
    d = datetime.now()
    dname = os.path.join("output", "timing", "%d-%d-%d" % (d.year, d.month, d.day))
    if not os.path.exists(dname):
        os.makedirs(dname)
    fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
    print("Writing to %s ..." % fname)

    # Output file
    header = ["repl", "method",  "n", "p_tr", "max_rank", "rank", "lbd", "time", "expl_var_val", "expl_var"]
    writer = csv.DictWriter(open(fname, "w", buffering=0),
                            fieldnames=header, quoting=csv.QUOTE_ALL)
    writer.writeheader()

    count = 0
    for repl, n in product(range(repeats), range_n):
        print("%s processing replicate %d for N=%d" % (str(datetime.now()), repl, n))

        # Generated data
        Ks_tr, K_tr, X_tr, X_te, y_tr, y_te = generate_data(n, max_rank, p_tr,
                                                            gamma_range=gamma_range, max_scales=max_scales)
        y_te = y_te.ravel()

        # Split test data in validation and holdout sets
        nt = len(y_te)
        X_val, y_val = X_te[:nt/2], y_te[:nt/2]
        X_hol, y_hol = X_te[nt/2:], y_te[nt/2:]

        # Output
        rows = []
        for method in methods:
            rank_range = range(2, max_rank, 2)
            for lbd, rank in product(lbd_range, rank_range):
                model = None
                try:
                    t1 = time()
                    if method == "Mklaren":
                        model = Mklaren(rank=rank, delta=delta, lbd=lbd)
                        model.fit(Ks_tr, y_tr)
                    elif method == "CSI":
                        model = RidgeLowRank(rank=rank, method="csi", lbd=lbd,
                                                 method_init_args={"delta": delta})
                        model.fit([K_tr], y_tr)
                    elif method == "ICD":
                        model = RidgeLowRank(rank=rank, method="icd", lbd=lbd)
                        model.fit([K_tr], y_tr)
                    elif method == "Nystrom":
                        model = RidgeLowRank(rank=rank, method="nystrom", lbd=lbd)
                        model.fit([K_tr], y_tr)
                    t = time() - t1
                except Exception as e:
                    print("%s error: %s" % (method, e.message))
                    continue

                # Evaluate result for a given rank
                if method == "Mklaren":
                    Xs_val = [X_val] * len(Ks_tr)
                    Xs_hol = [X_hol] * len(Ks_tr)
                else:
                    Xs_val = [X_val]
                    Xs_hol = [X_hol]
                yp_val = model.predict(Xs_val).ravel()
                yp_hol = model.predict(Xs_hol).ravel()

                evar_val = (np.var(y_val) - np.var(y_val - yp_val)) / np.var(y_val)
                evar_hol = (np.var(y_hol) - np.var(y_hol - yp_hol)) / np.var(y_hol)

                row = {"repl": repl, "method": method, "time": t,
                        "n": n, "rank": rank, "expl_var_val": evar_val, "lbd": lbd,
                       "expl_var": evar_hol, "max_rank": max_rank, "p_tr": p_tr}
                rows.append(row)

        # Write rows nevertheless
        count += len(rows)
        writer.writerows(rows)
        print("Written %d rows" % count)

if __name__ == "__main__":
    process2()
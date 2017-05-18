from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from time import time
from datetime import datetime
from itertools import product
import numpy as np
import csv
import os


def generate_data(n, max_rank, p_tr):
    """
    Generate data with a given number of inducing points within the training set.
    Compute signalusing the Nystrom approximation.
    :param n:
        Number of data points.
    :param max_rank:
        Number of inducing points.
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
    K = Kinterface(kernel=exponential_kernel, kernel_args={"gamma": 0.01}, data=X)
    K_tr = Kinterface(kernel=exponential_kernel, kernel_args={"gamma": 0.01}, data=X_tr)

    # Signal is defined using a random subset of the training set
    siginxs = np.random.choice(tr_inxs, size=max_rank, replace=False)
    alpha = np.zeros((n, 1))
    alpha[siginxs] = np.random.rand(len(siginxs), 1)

    # Signal is defined in the span of the training set
    # Use Nystrom approximation to compute signal efficiently
    Ka = K[siginxs, :].dot(alpha)
    KiKa = np.linalg.inv(K[siginxs, siginxs]).dot(Ka)
    y_true = K[:, siginxs].dot(KiKa)
    y_true = y_true - y_true.mean()
    y_tr = y_true[tr_inxs]
    y_te = y_true[te_inxs]

    return K_tr, X_tr, X_te, y_tr, y_te


def process():
    # Fixed hyper parameters
    repeats = 10                                                # Number of replicas.
    lbd = 0.0                                                   # Regularization parameter
    p_tr = 0.6                                                  # Fraction of test set
    range_n = map(int, 1.0/p_tr * np.array([1e5, 3e5, 1e6]))    # Number of samples in TRAINING set.
    methods = ["Mklaren", "ICD", "Nystrom"]                     # Methods
    delta = 10                                                  # Lookahead columns
    max_rank = 50                                               # Max. rank and number of indicuing points.

    # Create output directory
    d = datetime.now()
    dname = os.path.join("output", "timing", "%d-%d-%d" % (d.year, d.month, d.day))
    if not os.path.exists(dname):
        os.makedirs(dname)
    fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
    print("Writing to %s ..." % fname)

    # Output file
    header = ["repl", "method",  "n", "p_tr", "max_rank", "rank", "time", "expl_var"]
    writer = csv.DictWriter(open(fname, "w", buffering=0),
                            fieldnames=header, quoting=csv.QUOTE_ALL)
    writer.writeheader()

    count = 0
    for repl, n in product(range(repeats), range_n):
        print("%s processing replicate %d for N=%d" % (str(datetime.now()), repl, n))

        # Generated data
        K_tr, X_tr, X_te, y_tr, y_te = generate_data(n, max_rank, p_tr)
        y_te = y_te.ravel()

        # Output
        rows = []
        for method in methods:
            rank_range = range(10, max_rank, 10)
            for ri, rank in enumerate(rank_range):
                model = None
                try:
                    t1 = time()
                    if method == "Mklaren":
                        model = Mklaren(rank=rank, delta=delta, lbd=lbd)
                        model.fit([K_tr], y_tr)
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
                yp = model.predict([X_te]).ravel()
                evar = (np.var(y_te) - np.var(y_te - yp)) / np.var(y_te)
                row = {"repl": repl, "method": method, "time": t,
                        "n": n, "rank": rank, "expl_var": evar, "max_rank": max_rank, "p_tr": p_tr}
                rows.append(row)

        # Write rows nevertheless
        count += len(rows)
        writer.writerows(rows)
        print("Written %d rows" % count)

if __name__ == "__main__":
    process()
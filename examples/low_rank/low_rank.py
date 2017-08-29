hlp = """
    Evaluation of low-rank kernel approximation methods
    with ridge regression on standard datasets from KEEL.
"""

import os
import csv
import sys
import itertools as it
import scipy.stats as st
import time
import argparse

# Low-rank approximation methods
from mklaren.regression.ridge import RidgeLowRank
from mklaren.regression.ridge import RidgeMKL
from mklaren.regression.fitc import FITC
from mklaren.projection.rff import RFF
from mklaren.mkl.mklaren import Mklaren

# Kernels
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface

# Datasets
from datasets.keel import load_keel, KEEL_DATASETS

# Utils
from numpy import var, mean, logspace
from random import shuffle, seed
from math import ceil

# Comparable parameters with effective methods
N               = 3000                            # Max number of data points
rank_range      = range(2, 100)                   # Rang of tested ranks
p_range         = (10,)                           # Fixed number of kernels
lbd_range       = [0] + list(logspace(-5, 1, 7))  # Regularization parameter
delta           = 10                              # Number of look-ahead columns (CSI and mklaren)
cv_iter         = 5                               # Cross-validation iterations
training_size   = 0.6                             # Training set fraction
validation_size = 0.2                             # Validation set (for fitting hyperparameters)
test_size       = 0.2                             # Test set (for reporting scores)


# Method classes and fixed hyperparameters
methods = {
    "CSI" :        (RidgeLowRank, {"method": "csi", "method_init_args": {"delta": delta}}),
    "ICD" :        (RidgeLowRank, {"method": "icd"}),
    "Nystrom":     (RidgeLowRank, {"method": "nystrom"}),
    "CSI*" :       (RidgeLowRank, {"method": "csi", "method_init_args": {"delta": delta}}),
    "ICD*" :       (RidgeLowRank, {"method": "icd"}),
    "Nystrom*":    (RidgeLowRank, {"method": "nystrom"}),
    "Mklaren":     (Mklaren,      {"delta": delta}),
    "RFF":         (RFF,          {"delta": delta}),
    "FITC":        (FITC,         {}),
    "uniform":     (RidgeMKL,     {"method": "uniform"}),
    "L2KRR":       (RidgeMKL,     {"method": "l2krr"}),
}


def process(dataset, outdir):
    """
    Run experiments with specified parameters.
    :param dataset: Dataset key.
    :param outdir: Output directory.
    :return:
    """

    # Fixed output
    # Create output directory
    if not os.path.exists(outdir): os.makedirs(outdir)
    fname = os.path.join(outdir, "%s.csv" % dataset)

    # Output
    header = ["dataset", "n", "method", "rank", "erank", "iteration", "lambda",
              "gmin", "gmax", "p", "evar_tr", "evar", "time",
              "RMSE_tr", "RMSE_va", "RMSE"]
    fp = open(fname, "w", buffering=0)
    writer = csv.DictWriter(fp, fieldnames=header)
    writer.writeheader()

    # Load data
    data = load_keel(n=N, name=dataset)

    # Load data and normalize columns
    X = st.zscore(data["data"], axis=0)
    y = st.zscore(data["target"])
    n = len(X)

    # Perform model cross-validation with internal parameter selection
    for cv, p in it.product(range(cv_iter), p_range):
        gam_range = logspace(-6, 3, p, base=2)

        # Split into training, validation, test sets.
        seed(cv)
        inxs = range(len(X))
        shuffle(inxs)
        n1 = int(training_size * n)
        n2 = int(validation_size * n)
        tr, tval, te = inxs[:n1], inxs[n1:n1+n2], inxs[n1+n2:]
        X_tr, X_val, X_te = X[tr], X[tval], X[te]
        y_tr, y_val, y_te = y[tr], y[tval], y[te]

        # Store kernel interfaces for training data
        Ks_full = [Kinterface(data=X,
                         kernel=exponential_kernel,
                         kernel_args={"gamma": gam}) for gam in gam_range]
        Ks = [Kinterface(data=X_tr,
                         kernel=exponential_kernel,
                         kernel_args={"gamma": gam}) for gam in gam_range]
        Ksum = Kinterface(data=X_tr,
                          kernel=kernel_sum,
                          kernel_args={"kernels": [exponential_kernel] * len(gam_range),
                                       "kernels_args": [{"gamma": gam} for gam in gam_range]})

        for mname, (Mclass, kwargs) in methods.items():

            # Fit hyperparameters on the validation set using the current rank
            for rank in rank_range:
                for lbd in lbd_range:
                    times = []
                    try:
                        t_train = time.time()
                        if mname == "Mklaren":
                            effective_rank = rank
                            model = Mclass(lbd=lbd, rank=rank, **kwargs)
                            model.fit(Ks, y_tr)
                            yptr    = model.predict([X_tr] * len(Ks)).ravel()
                            ypva    = model.predict([X_val] * len(Ks)).ravel()
                            ypte    = model.predict([X_te] * len(Ks)).ravel()
                        elif mname == "RFF":
                            effective_rank = rank
                            model = Mclass(rank=rank, lbd=lbd,
                                           gamma_range=gam_range, **kwargs)
                            model.fit(X_tr, y_tr)
                            yptr = model.predict(X_tr).ravel()
                            ypva = model.predict(X_val).ravel()
                            ypte = model.predict(X_te).ravel()
                        elif mname == "FITC":
                            effective_rank = rank
                            model = Mclass(rank=rank, **kwargs)
                            model.fit(Ks, y_tr)
                            yptr = model.predict([X_tr] * len(Ks)).ravel()
                            ypva = model.predict([X_val]* len(Ks)).ravel()
                            ypte = model.predict([X_te] * len(Ks)).ravel()
                        elif mname in ("uniform", "L2KRR"):
                            effective_rank = rank
                            model = Mclass(lbd=lbd, **kwargs)
                            model.fit(Ks_full, y, holdout=te+tval)
                            yptr = model.predict(tr).ravel()
                            ypva = model.predict(tval).ravel()
                            ypte = model.predict(te).ravel()
                        elif mname in ("CSI*", "ICD*", "Nystrom*"):   # Separate approximations
                            effective_rank = int(max(1, ceil(1.0 * rank / p)))
                            model = Mclass(lbd=lbd, rank=effective_rank, **kwargs)
                            model.fit(Ks, y_tr)
                            yptr = model.predict([X_tr] * len(Ks)).ravel()
                            ypva = model.predict([X_val] * len(Ks)).ravel()
                            ypte = model.predict([X_te] * len(Ks)).ravel()
                        else:   # Other low-rank approximations; Mklaren2
                            effective_rank = rank
                            model = Mclass(lbd=lbd, rank=rank, **kwargs)
                            model.fit([Ksum], y_tr)
                            yptr = model.predict([X_tr]).ravel()
                            ypva = model.predict([X_val]).ravel()
                            ypte = model.predict([X_te]).ravel()
                        t_train = time.time() - t_train
                        times.append(t_train)
                    except Exception as e:
                        sys.stderr.write("Method: %s rank: %d iter: %d error: %s \n" % (mname, rank, cv, str(e)))
                        continue

                    # Compute errors
                    score_tr = var(y_tr - yptr) ** 0.5
                    score_va = var(y_val - ypva) ** 0.5
                    score_te = var(y_te - ypte) ** 0.5

                    # Explained variance
                    evar_tr = (var(y_tr) - var(y_tr - yptr)) / var(y_tr)
                    evar_te = (var(y_te) - var(y_te - ypte)) / var(y_te)

                    # Write to output
                    row = {"dataset": dataset, "method": mname, "rank": rank, "n": n,
                           "iteration": cv, "lambda": lbd, "time": mean(times),
                           "evar": evar_te, "evar_tr": evar_tr, "erank": effective_rank,
                           "RMSE": score_te, "RMSE_va": score_va, "RMSE_tr": score_tr,
                           "gmin": min(gam_range), "gmax": max(gam_range), "p": len(gam_range)}
                    writer.writerow(row)

                    # Break for FITC / no lambda
                    if mname == "FITC": break

                # Break for FITC / no rank
                if mname in ("uniform", "L2KRR"): break


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description=hlp)
    parser.add_argument("dataset", help="Dataset. One of {%s}." % ", ".join(KEEL_DATASETS))
    parser.add_argument("output",  help="Output directory.")
    args = parser.parse_args()

    # Output directory
    data_set = args.dataset
    out_dir = args.output
    assert data_set in KEEL_DATASETS
    process(data_set, out_dir)
"""
    Evaluation of low-rank kernel approximation methods
    with ridge regression on standard datasets.
"""
import os
import csv
import datetime
import sys
import itertools as it

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
from datasets.delve import load_abalone
from datasets.delve import load_boston
from datasets.delve import load_comp_activ
from datasets.delve import load_bank
from datasets.delve import load_pumadyn
from datasets.delve import load_kin
from datasets.delve import load_census_house
from datasets.orange import load_ionosphere
from datasets.keel import load_keel

# Utils
from numpy import var, mean, logspace, where
from numpy.linalg import norm
from random import shuffle, seed

# Datasets and options
# Load max. 1000 examples
n    = 1000
dset = dict(enumerate(sys.argv)).get(1, "boston")
dset_sub = dict(enumerate(sys.argv)).get(2, None)

# Load datasets with at most n examples
datasets = {
    "boston":    (load_boston,       {"n": n,}),
    "ionosphere": (load_ionosphere, {"n": n,}),
    "abalone":   (load_abalone,      {"n": n,}),
    "comp":      (load_comp_activ,   {"n": n,}),
    "bank":      (load_bank,         {"typ": "8fm", "n": n,}),
    "pumadyn":   (load_pumadyn,      {"typ": "8fm", "n": n,}),
    "kin":       (load_kin,          {"typ": "8fm", "n": n,}),
    "census":    (load_census_house, {"n": n,}),
    "keel":      (load_keel,         {"n": n, "name": dset_sub})
}


# Hyperparameters
rank_range = range(2, 21) + range(20, 85, 5) # Rank range
lbd_range  = [0] + list(logspace(-5, 1, 7))  # Regularization parameter
delta      = 10                              # Number of look-ahead columns (CSI and mklaren)
p_range    = (1, 2, 3, 5, 10, 30)

# Method classes and fixed hyperparameters
methods = {
    "CSI" :        (RidgeLowRank, {"method": "csi",
                                   "method_init_args": {"delta": delta}}),
    "ICD" :        (RidgeLowRank, {"method": "icd"}),
    "Nystrom":     (RidgeLowRank, {"method": "nystrom"}),
    "Mklaren":     (Mklaren,      {"delta": delta}),
    "Mklaren2":    (Mklaren,      {"delta": delta}),
    "RFF":         (RFF,          {"delta": delta}),
    "FITC":        (FITC, {}),
    # "uniform":     (RidgeMKL,     {"method": "uniform"}),
    "L2KRR":       (RidgeMKL,     {"method": "l2krr"}),
}


# CV hyper parameters
cv_iter         = 5     # Cross-validation iterations
training_size   = 0.6   # Training set
validation_size = 0.2   # Validation set (for fitting hyperparameters)
test_size       = 0.2   # Test set (for reporting scores)


# Fixed output
# Create output directory
d = datetime.datetime.now()
dname = os.path.join("..", "output", "delve_regression",
                     "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname): os.makedirs(dname)
rcnt = len(os.listdir(dname))
fname = os.path.join(dname, "results_%d.csv" % rcnt)

# Output
header = ["dataset", "method", "rank", "iteration", "lambda",
          "gmin", "gmax", "p",
          "RMSE_tr", "RMSE_va", "RMSE"]
fp = open(fname, "w", buffering=0)
writer = csv.DictWriter(fp, fieldnames=header)
writer.writeheader()

# Load data
load, load_args = datasets[dset]
data = load(**load_args)
if dset_sub is not None: dset = dset_sub

# Load data and normalize columns
X = data["data"]
X = X - X.mean(axis=0)
nrm = norm(X, axis=0)
nrm[where(nrm==0)] = 1
X /=  nrm
y = data["target"]
y -= mean(y)
n = len(X)


# Perform model cross-validation with internal parameter selection
for cv, p in it.product(range(cv_iter), p_range):
    gam_range = logspace(-6, 6, p, base=2)  # RBF kernel parameter

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
                try:
                    if mname == "Mklaren":
                        model = Mclass(lbd=lbd, rank=rank, **kwargs)
                        model.fit(Ks, y_tr)
                        yptr    = model.predict([X_tr for k in Ks])
                        ypva    = model.predict([X_val for k in Ks])
                        ypte    = model.predict([X_te for k in Ks])
                    elif mname == "RFF":
                        model = Mclass(rank=rank, lbd=lbd,
                                       gamma_range=gam_range, **kwargs)
                        model.fit(X_tr, y_tr)
                        yptr = model.predict(X_tr)
                        ypva = model.predict(X_val)
                        ypte = model.predict(X_te)
                    elif mname == "FITC":
                        model = Mclass(rank=rank, **kwargs)
                        model.fit(Ks, y_tr)
                        yptr = model.predict([X_tr for k in Ks])
                        ypva = model.predict([X_val for k in Ks])
                        ypte = model.predict([X_te for k in Ks])
                    elif mname in ("uniform", "L2KRR"):
                        model = Mclass(lbd=lbd, **kwargs)
                        model.fit(Ks_full, y, holdout=te+tval)
                        yptr = model.predict(tr)
                        ypva = model.predict(tval)
                        ypte = model.predict(te)
                    else:   # Other low-rank approximations
                        model = Mclass(lbd=lbd, rank=rank, **kwargs)
                        model.fit([Ksum], y_tr)
                        yptr = model.predict([X_tr])
                        ypva = model.predict([X_val])
                        ypte = model.predict([X_te])
                except Exception as e:
                    sys.stderr.write("Method: %s rank: %d iter: %d error: %s \n" % (mname, rank, cv, str(e)))
                    continue

                # Compute errors
                score_tr = var(y_tr - yptr) ** 0.5
                score_va = var(y_val - ypva) ** 0.5
                score_te = var(y_te - ypte) ** 0.5

                # Write to output
                row = {"dataset": dset, "method": mname, "rank": rank,
                       "iteration": cv, "lambda": lbd,
                       "RMSE": score_te, "RMSE_va": score_va, "RMSE_tr": score_tr,
                       "gmin": min(gam_range), "gmax": max(gam_range), "p": len(gam_range)}
                writer.writerow(row)

                # Break for FITC / no lambda
                if mname == "FITC": break

            # Break for FITC / no rank
            if mname in ("uniform", "L2KRR"): break
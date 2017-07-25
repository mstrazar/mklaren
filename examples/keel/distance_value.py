"""
    Evaluation of low-rank kernel approximation methods
    with ridge regression on standard datasets.
"""
# Kernels
import os
import numpy as np
import csv
import scipy.spatial as spt
import scipy.stats as st
import matplotlib.pyplot as plt
import itertools as it
from sklearn.manifold.mds import MDS
from mklaren.mkl.mklaren import Mklaren
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface

def estimate_rel_error(X, Z, y):
    """
    Estimate a relative error in norm.
    :param X: Original feature space.
    :param Z: Approximated feature space.
    :param y: Target values.
    :return: Mean relative error.
    """
    n = X.shape[0]
    nc = n * (n - 1) / 2
    errors = np.zeros((nc, ))
    x_dists = np.zeros((nc, ))
    z_dists = np.zeros((nc,))
    y_dists = np.zeros((nc,))
    c = 0
    for i, j in it.combinations(range(n), 2):
        a = np.linalg.norm(X[i, :] - X[j, :])
        b = np.linalg.norm(Z[i, :] - Z[j, :])
        errors[c] = abs(a - b) / a
        x_dists[c] = a
        z_dists[c] = b
        y_dists[c] = abs(y[i] - y[j])
        c += 1
    return np.mean(errors), x_dists, z_dists, y_dists


# Datasets
from datasets.keel import load_keel, KEEL_DATASETS

# Datasets and options
# Load max. 1000 examples
outdir = "/Users/martin/Dev/mklaren/examples/output/keel/distances/"
n    = 1000
rank_range = [2, 5, 10, 30, 100]
delta_range = [2, 5, 10, 30]
gam_range = np.logspace(-6, 6, 5)

# Write results to csv
header = ["dataset", "D", "n", "rank", "delta", "dist.val.corr", "evar"]
rname = os.path.join(outdir, "results.csv")
fp = open(rname, "w", buffering=0)
writer = csv.DictWriter(fp, fieldnames=header)
writer.writeheader()

for dset_sub in KEEL_DATASETS:
    # Load data
    data = load_keel(name=dset_sub, n=n)

    # Compute pairwise distances
    X = data["data"]
    y = st.zscore(data["target"])
    D = spt.distance_matrix(X, X)

    # Average distance
    dists = np.array([d for d in D.ravel() if d != 0])
    dists = dists / dists.max()

    # Fit Mklaren
    Ks = [Kinterface(data=X,
                     kernel=exponential_kernel,
                     kernel_args={"gamma": gam}) for gam in gam_range]

    for rank, delta in it.product(rank_range, delta_range):
        mklaren = Mklaren(rank=rank, delta=delta, lbd=0.0)
        try:
            mklaren.fit(Ks, y)
        except Exception as e:
            print(e)
            continue
        inxs = set().union(*[set(mklaren.data[i]["act"])
                             for i in range(len(gam_range))])

        # Explained variance
        yp = mklaren.predict([X] * len(gam_range))
        evar = (np.var(y) - np.var(y-yp)) / np.var(y)

        # Get distance from distance pairs
        dp, xd, _, yd = estimate_rel_error(X, X, y)
        dv_corr = st.spearmanr(xd, yd)[0]

        # Write results
        row  = {"dataset": dset_sub, "D": X.shape[1], "n": X.shape[0],
                "dist.val.corr": dv_corr, "evar": evar, "rank": rank, "delta": delta}
        writer.writerow(row)


# End
fp.close()
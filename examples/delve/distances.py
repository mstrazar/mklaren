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
outdir = "../output/delve_regression/distances/"
n    = 1000

# Write results to csv
header = ["dataset", "dp", "D", "n", "dist.val.corr"]
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
    y = y + y.min()
    D = spt.distance_matrix(X, X)

    # Average distance
    dists = np.array([d for d in D.ravel() if d != 0])
    dists = dists / dists.max()

    # Nearest neightbour distance
    Dm = D.copy()
    np.fill_diagonal(Dm, np.inf)
    nn_dists = Dm.min(axis=0).ravel()

    # Fit Mklaren
    gam_range = np.logspace(-6, 6, 5)
    Ks = [Kinterface(data=X,
                     kernel=exponential_kernel,
                     kernel_args={"gamma": gam}) for gam in gam_range]
    mklaren = Mklaren(rank=10, delta=10, lbd=0.01)
    try:
        mklaren.fit(Ks, y)
    except Exception as e:
        print(e)
        continue
    inxs = set().union(*[set(mklaren.data[i]["act"])
                         for i in range(len(gam_range))])


    # Fit MDS (2D)
    model = MDS(n_components=2)
    try:
        Z = model.fit_transform(X)
        dp, xd, zd, yd = estimate_rel_error(X, Z, y)
    except ValueError as e:
        print(e)
        continue

    # Correation between distance and value
    dv_corr = st.spearmanr(xd, yd)[0]

    fname = os.path.join(outdir, "mds2D_%s.pdf" % dset_sub)
    plt.figure()
    for i in range(X.shape[0]):
        if i in inxs:
            plt.plot(Z[i, 0], Z[i, 1], "^", markersize=5 * y[i], alpha=0.8, color="red")
        else:
            plt.plot(Z[i, 0], Z[i, 1], "k.", markersize=5 * y[i], alpha=0.1)
    plt.title("%s (%d-D %.3f %%)" % (dset_sub, X.shape[1], 100*dp))
    plt.xlabel("$Z_1$")
    plt.ylabel("$Z_2$")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    # Plot 1D
    fname = os.path.join(outdir, "mds1D_%s.pdf" % dset_sub)
    plt.figure()
    for i in range(X.shape[0]):
        if i in inxs:
            plt.plot(Z[i, 0], y[i], "^", alpha=0.8, color="red", markersize=5)
        else:
            plt.plot(Z[i, 0], y[i], "k.", alpha=0.1)
    plt.title("%s (%d-D)" % (dset_sub, X.shape[1]))
    plt.xlabel("$Z_1$")
    plt.ylabel("$y$")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    for name, dst in zip(("dists", "nn_dists"), (dists, nn_dists)):
        # Draw histogram
        fname = os.path.join(outdir, "%s_%s.pdf" % (name, dset_sub))
        plt.figure()
        plt.hist(dst, bins=30, edgecolor="none",  color="gray")
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.title(dset_sub)
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print "Written %s %d %.3f" % (dset_sub, X.shape[1], dp)

    row  = {"dataset": dset_sub,
            "dp": dp, "D": X.shape[1], "n": X.shape[0],
            "dist.val.corr": dv_corr}
    writer.writerow(row)


# End
fp.close()
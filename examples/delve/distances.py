"""
    Evaluation of low-rank kernel approximation methods
    with ridge regression on standard datasets.
"""
# Kernels
import os
import numpy as np
import scipy.spatial as spt
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.manifold.mds import MDS

# Datasets
from datasets.keel import load_keel, KEEL_DATASETS

# Datasets and options
# Load max. 1000 examples
outdir = "../output/delve_regression/distances/"
n    = 1000

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

    # Fit MDS (2D)
    model = MDS(n_components=2)
    try:
        Z = model.fit_transform(X)
    except ValueError as e:
        print(e)
        continue

    fname = os.path.join(outdir, "mds2D_%s.pdf" % dset_sub)
    plt.figure()
    for i in range(X.shape[0]):
        plt.plot(Z[i, 0], Z[i, 1], "k.", markersize=5 * y[i], alpha=0.1)
    plt.title("%s (%d-D)" % (dset_sub, X.shape[1]))
    plt.xlabel("$Z_1$")
    plt.ylabel("$Z_2$")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()

    # Fit MDS (1D)
    model = MDS(n_components=1)
    try:
        Z = model.fit_transform(X)
    except ValueError as e:
        print(e)
        continue

    fname = os.path.join(outdir, "mds1D_%s.pdf" % dset_sub)
    plt.figure()
    for i in range(X.shape[0]):
        plt.plot(Z[i], y[i], "k.", alpha=0.1)
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
        print "Written %s (%d)" % (fname, X.shape[1])
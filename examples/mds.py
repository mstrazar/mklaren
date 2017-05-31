import matplotlib as mpl
mpl.use("Agg")

import csv
import os
import matplotlib.pyplot as plt
import GPy
import numpy as np
import itertools as it
from sklearn.manifold.mds import MDS
from datasets.delve import *


# Store images only
outdir = "output/mds/images/"
outlog = "output/mds/results.csv"

# Load datasets with at most n examples
n = 1000
datasets = {
    "census":    (load_census_house,     {"n": n,}),
    "boston":    (load_boston,     {"n": n,}),
    "abalone":   (load_abalone,    {"n": n,}),
    "comp":      (load_comp_activ, {"n": n,}),
    "bank":      (load_bank, {"typ": "8fm", "n": n,}),
    "pumadyn":   (load_pumadyn, {"typ": "8fm", "n": n,}),
    "kin":       (load_kin, {"typ": "8fm", "n": n,}),
}


# Write to log
header = ["dataset", "error"]
writer = csv.DictWriter(open(outlog, "w"), fieldnames=header)
writer.writeheader()

def estimate_rel_error(X, Z):
    """
    Estimate a relative error in norm.
    :param X: Original feature space.
    :param Z: Approximated feature space.
    :return: Mean relative error.
    """
    n = X.shape[0]
    nc = n * (n - 1) / 2
    errors = np.zeros((nc, ))
    c = 0
    for i, j in it.combinations(range(n), 2):
        a = np.linalg.norm(X[i, :] - X[j, :])
        b = np.linalg.norm(Z[i, :] - Z[j, :])
        errors[c] = abs(a - b) / a
    return np.mean(errors)


for dset, (load, load_args) in datasets.iteritems():

    # Load data and approximate with MDS
    data = load(**load_args)
    X, y = data["data"], data["target"]
    y = y.reshape((len(y), 1)) - y.mean()
    model = MDS(n_components=1)
    Z = model.fit_transform(X, y)
    inxs = np.argsort(Z.ravel())
    Z = Z[inxs, :]
    y = y[inxs, :]
    re = estimate_rel_error(X, Z)

    # Optimize a GP with exponentiated quadratic kernel
    k = GPy.kern.ExpQuad(1, lengthscale=np.std(Z))
    m = GPy.models.GPRegression(Z, y, kernel=k)
    m.optimize('bfgs', max_iters=1000)
    _, var = m.predict(Z)

    # Draw figures
    fname = os.path.join(outdir, "mds_%s.pdf" % dset)
    plt.figure(figsize=(6, 4))
    plt.title("{0} (rel. err. {1} %)".format(dset, "%.4f" % (100 * re)) )
    m.plot(ax=plt.gca())
    plt.xlabel("MDS (1D) approx. to input space")
    plt.ylabel("Target variable (y)")
    plt.legend()
    plt.savefig(fname)
    print("Written %s" % fname)

    fname = os.path.join(outdir, "var_%s.pdf" % dset)
    plt.figure(figsize=(6, 2))
    plt.plot(Z.ravel(), var, "k-")
    plt.xlabel("MDS (1D) approx. to input space")
    plt.ylabel("Fitted noise variance")
    plt.savefig(fname)
    print("Written %s" % fname)

    # Write to log
    writer.writerow({"dataset": dset, "error": re})


"""
    Evaluation of low-rank kernel approximation methods
    with ridge regression on standard datasets.
"""
# Kernels
import sys
import os
import csv
import numpy as np
import scipy.stats as st
import itertools as it
import datetime
from sklearn.manifold.mds import MDS
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface
from datasets.keel import load_keel

from mklaren.mkl.mklaren import Mklaren
from mklaren.projection.rff import RFF
from mklaren.regression.ridge import RidgeLowRank
from mklaren.regression.fitc import FITC

# Choose dataset
dset_sub = dict(enumerate(sys.argv)).get(1, "abalone")

# Datasets and options
# Load max. 1000 examples
n    = 2000
p_tr = 0.6
p_va = 0.2
delta = 10
plot = True
gam_range = np.logspace(-6, 6, 13, base=2)
deg_range = range(5)
lbd_range = list(np.logspace(-5, 1, 7)) + [0]
dim_range = [2, 5, 10]
rank_range = [5, 10, 30]
cv_range = range(10)
meths = ["Mklaren", "CSI", "FITC", "RFF"]

# Fixed output
# Create output directory
d = datetime.datetime.now()
dname = os.path.join("..", "output", "delve_regression", "distances_cv_nonrandom",
                     "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname): os.makedirs(dname)
fname = os.path.join(dname, "results_%s.csv" % dset_sub)

# Output
header = ["dataset", "n", "method", "rank", "iteration", "lambda", "D",
          "p", "evar_tr", "evar_va", "evar", "corr", "corr.p", "dcorr", "dcorr.p"]
fp = open(fname, "w", buffering=0)
writer = csv.DictWriter(fp, fieldnames=header)
writer.writeheader()

# Kernels
kernels = []
kernels.extend([(exponential_kernel, {"gamma": g}) for g in gam_range])

data = load_keel(name=dset_sub, n=n)
X = data["data"]
X = X - X.mean(axis=0)
nrm = np.linalg.norm(X, axis=0)
nrm[np.where(nrm == 0)] = 1
X /= nrm
y = st.zscore(data["target"])

# Deduce number of training samples
n_tr = int(p_tr * X.shape[0])
n_va = int(p_va * X.shape[0])

# CV for multiple settings
for cv, rank, D in it.product(cv_range, rank_range, dim_range):
    np.random.seed(cv)

    # Fit MDS (2D)
    model = MDS(n_components=D, random_state=cv)
    Z = model.fit_transform(X)

    # Define training and test set
    center = Z.mean(axis=0)
    distance = np.sqrt(np.power(Z - center, 2).sum(axis=1))
    rnkd = st.rankdata(distance)

    # Non-random sampling
    tr = np.where(rnkd <= n_tr)[0]
    va = np.where(np.logical_and(rnkd > n_tr, rnkd <= n_tr + n_va))[0]
    te = np.where(rnkd > n_va + n_tr)[0]

    # Random sampling
    # jnx = np.arange(len(Z), dtype=int)
    # np.random.shuffle(jnx)
    # tr = jnx[:n_tr]
    # va = jnx[n_tr:n_tr+n_va]
    # te = jnx[n_tr+n_va:]

    # Fit methods on Z
    for method in meths:
        Ks = [Kinterface(data=Z[tr],
                         row_normalize=True,
                         kernel=kern[0],
                         kernel_args=kern[1]) for kern in kernels]
        Ksum = Kinterface(data=Z[tr],
                          row_normalize=True,
                          kernel=kernel_sum,
                          kernel_args={"kernels":      [kern[0] for kern in kernels],
                                       "kernels_args": [kern[1] for kern in kernels]})

        Yt = Yv = Yp = None
        for lbd in lbd_range:
            if method == "Mklaren":
                mklaren = Mklaren(rank=rank, delta=delta, lbd=lbd)
                try:
                    mklaren.fit(Ks, y[tr])
                except Exception as e:
                    print(e)
                    continue
                inxs = set().union(*[set(mklaren.data[i]["act"])
                                     for i in range(len(gam_range))])
                inxs = tr[list(inxs)]
                Yt = mklaren.predict([Z[tr] for g in gam_range])
                Yv = mklaren.predict([Z[va] for g in gam_range])
                Yp = mklaren.predict([Z[te] for g in gam_range])
            elif method == "FITC":
                model = FITC(rank=rank)
                model.fit(Ks, y[tr])
                Yt = model.predict([Z[tr] for k in Ks]).ravel()
                Yv = model.predict([Z[va] for k in Ks]).ravel()
                Yp = model.predict([Z[te] for k in Ks]).ravel()
                inxs = [np.argmin(np.power(Z[tr] - a, 2).sum(axis=1)) for a in model.anchors_]
                inxs = tr[list(inxs)]
            elif method == "CSI":
                ridge = RidgeLowRank(rank=rank,
                                     method_init_args={"delta": delta},
                                     method="csi", lbd=lbd)
                try:
                    ridge.fit([Ksum], y[tr])
                except Exception as e:
                    print(e)
                    continue
                Yt = ridge.predict([Z[tr] for g in gam_range])
                Yv = ridge.predict([Z[va] for g in gam_range])
                Yp = ridge.predict([Z[te] for g in gam_range])
                inxs = set().union(*map(set, ridge.active_set_))
                inxs = tr[list(inxs)]
            elif method == "RFF":
                rff = RFF(rank=10, delta=10, gamma_range=gam_range, lbd=0.01)
                rff.fit(Z[tr], y[tr])
                Yt = rff.predict(Z[tr])
                Yv = rff.predict(Z[va])
                Yp = rff.predict(Z[te])
                inxs = set()

            # Explained variance
            evar_tr = (np.var(y[tr]) - np.var(y[tr] - Yt)) / np.var(y[tr])
            evar_va = (np.var(y[va]) - np.var(y[va] - Yv)) / np.var(y[va])
            evar    = (np.var(y[te]) - np.var(y[te] - Yp)) / np.var(y[te])
            mse_tr  = np.power(y[tr] - Yt, 2)
            mse     = np.power(y[te] - Yp, 2)
            dr, drho = st.pearsonr(distance[te], mse)

            # Fit to data
            pr, prho = st.pearsonr(Yp.ravel(), y[te])

            # Write results
            row = {"dataset": dset_sub, "n": Z.shape[0] , "method": method,
                   "rank": rank, "iteration": cv, "lambda": lbd,
                   "p": len(Ks), "evar_tr": evar_tr, "evar_va": evar_va, "evar": evar,
                   "corr": pr, "corr.p": prho,
                   "dcorr": dr, "dcorr.p": drho, "D": D}
            writer.writerow(row)

            # Break lambda
            if method == "FITC":
                break
"""
    Evaluation of low-rank kernel approximation methods
    with ridge regression on standard datasets.
"""
# Kernels
import os
import numpy as np
import scipy.stats as st
import itertools as it
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.manifold.mds import MDS
from mklaren.kernel.kernel import exponential_kernel, kernel_sum, periodic_kernel
from mklaren.kernel.kinterface import Kinterface
from datasets.keel import load_keel, KEEL_DATASETS

from mklaren.mkl.mklaren import Mklaren
from mklaren.projection.rff import RFF
from mklaren.regression.ridge import RidgeLowRank

# Datasets and options
# Load max. 1000 examples
outdir = "../output/delve_regression/distances/"
n    = 300
gam_range = np.logspace(-6, 6, 1, base=2)
meths = ["Mklaren", "CSI", "RFF", ]

# Kernels
kernels = []
kernels.extend([(exponential_kernel, {"gamma": g}) for g in gam_range])
# kernels.extend([(periodic_kernel, {"l": g}) for g in gam_range])

for dset_sub in KEEL_DATASETS:
    # Load data
    data = load_keel(name=dset_sub, n=n)
    X = data["data"]
    X = X - X.mean(axis=0)
    nrm = np.linalg.norm(X, axis=0)
    nrm[np.where(nrm == 0)] = 1
    X /= nrm
    y = st.zscore(data["target"])
    y -= y.min()

    # Fit MDS (2D)
    model = MDS(n_components=2)
    try:
        Z = model.fit_transform(X)
        zxa, zya = np.min(Z, axis=0)
        zxb, zyb = np.max(Z, axis=0)
        Zp = np.array(list(it.product(np.linspace(zxa, zxb, 100),
                                      np.linspace(zya, zyb, 100))))
        zx = Zp[:,0].reshape((100, 100))
        zy = Zp[:,1].reshape((100, 100))
    except ValueError as e:
        print(e)
        continue

    # Fit methods on Z
    for method in meths:
        if method == "Mklaren":
            Ks = [Kinterface(data=Z,
                             kernel=kern[0],
                             kernel_args=kern[1]) for kern in kernels]
            mklaren = Mklaren(rank=10, delta=10, lbd=0.01)
            try:
                mklaren.fit(Ks, y)
            except Exception as e:
                print(e)
                continue
            inxs = set().union(*[set(mklaren.data[i]["act"])
                                 for i in range(len(gam_range))])
            Yt = mklaren.predict([Z for g in gam_range])
            Yp = mklaren.predict([Zp for g in gam_range])
        elif method == "CSI":
            Ksum = Kinterface(data=Z,
                              kernel=kernel_sum,
                              kernel_args={"kernels": [kern[0] for kern in kernels],
                                           "kernels_args": [kern[1] for kern in kernels]})
            ridge = RidgeLowRank(rank=10,
                                 method_init_args={"delta": 10},
                                 method="csi", lbd=0.01)
            try:
                ridge.fit([Ksum], y)
            except Exception as e:
                print(e)
                continue
            Yt = ridge.predict([Z for g in gam_range])
            Yp = ridge.predict([Zp for g in gam_range])
            inxs = set().union(*map(set, ridge.active_set_))
        elif method == "RFF":
            rff = RFF(rank=10, delta=10, gamma_range=gam_range, lbd=0.01)
            rff.fit(Z, y)
            Yt = rff.predict(Z)
            Yp = rff.predict(Zp)
            inxs = set()

        # Fit to data
        pr, prho = st.pearsonr(Yt.ravel(), y)
        print("Dataset: %s method: %s pr: %.3f (p = %.5f)" % (dset_sub, method, pr, prho))

        # Predicted values
        Yz = Yp.reshape((100, 100))

        # Plot a scatter
        fname = os.path.join(outdir, "mdsZ_%s_%s.pdf" % (dset_sub, method))
        plt.figure()
        levels = MaxNLocator(nbins=100).tick_values(Yp.min(), Yp.max())
        plt.contourf(zx, zy, Yz, cmap=plt.get_cmap('PiYG'), levels=levels)
        for i in range(X.shape[0]):
            if i in inxs:
                plt.plot(Z[i, 0], Z[i, 1], "^", markersize=5 * y[i], alpha=0.8, color="red")
            else:
                plt.plot(Z[i, 0], Z[i, 1], "k.", markersize=5 * y[i], alpha=0.1)
        plt.title("%s/%s (%d-D)" % (dset_sub, method, X.shape[1]))
        plt.xlabel("$Z_1$")
        plt.ylabel("$Z_2$")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print "Written %s" % fname
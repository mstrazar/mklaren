hlp = """
    Trajectory of grpup LARS with different penalty functions 
    and KMP greedy algorithm.
"""

#

import scipy.stats as st
import numpy as np
import os
os.environ["OCTAVE_EXECUTABLE"] = "/usr/local/bin/octave"

# Kernels
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface
from mklaren.regression.ridge import RidgeLowRank
from mklaren.mkl.kmp import KMP
from mklaren.regression.ridge import RidgeMKL

# Datasets
from datasets.keel import load_keel, KEEL_DATASETS

# Utils
import matplotlib.pyplot as plt

# New methods
from examples.lars.lars_mkl import LarsMKL
from examples.lars.lars_group import p_ri, p_const, p_sc, colors


# Parameters
out_dir = "/Users/martins/Dev/mklaren/examples/mkl/output/"
N = 2000
delta = 5
gamma = .1
p_tr = .8
lbd = 0.000
rank = 30

formats = {"lars-ri": "gv-",
           "lars-sc": "bv-",
           "lars-co": "cv-",
           "kmp": "c--",
           "icd": "b--",
           "nystrom": "m--",
           "csi": "r--",
           "L2KRR": "k-"}


def process(dataset):
    # Load data
    data = load_keel(n=N, name=dataset)
    gamma_range = np.logspace(-5, 5, 10)

    # Load data and normalize columns
    X = st.zscore(data["data"], axis=0)
    y = st.zscore(data["target"])
    inxs = np.argsort(y).ravel()
    X = X[inxs, :]
    y = y[inxs]

    # Training/test
    n = len(X)
    np.random.seed(42)
    tr = np.random.choice(range(n), size=int(p_tr * n), replace=False)
    te = np.array(list(set(range(n)) - set(tr)))
    tr = tr[np.argsort(y[tr].ravel())]
    te = te[np.argsort(y[te].ravel())]

    # Training kernels
    Ks_tr = [Kinterface(data=X[tr],
                        kernel=exponential_kernel,
                        kernel_args={"gamma": gamma})
             for gamma in gamma_range]
    Ks = [Kinterface(data=X,
                     kernel=exponential_kernel,
                     kernel_args={"gamma": gamma})
             for gamma in gamma_range]

    Ksum_tr = Kinterface(data=X[tr],
                      kernel=kernel_sum,
                      kernel_args={"kernels": [exponential_kernel] * len(gamma_range),
                                   "kernels_args": [{"gamma": gam} for gam in gamma_range]})

    # Collect test error paths
    results = dict()
    for m in formats.keys():
        if m == "lars-ri":
            model = LarsMKL(delta=delta, rank=rank, f=p_ri)
            model.fit(Ks_tr, y[tr])
            ypath = model.predict_path_ls([X[te]] * len(Ks_tr))
        elif m == "lars-ri-fast":
            model = LarsMKL(delta=delta, rank=rank, f=p_ri_fast)
            model.fit(Ks_tr, y[tr])
            ypath = model.predict_path_ls([X[te]] * len(Ks_tr))
        elif m == "lars-sc":
            model = LarsMKL(delta=delta, rank=rank, f=p_sc)
            model.fit(Ks_tr, y[tr])
            ypath = model.predict_path_ls([X[te]] * len(Ks_tr))
        elif m == "lars-co":
            model = LarsMKL(delta=delta, rank=rank, f=p_const)
            model.fit(Ks_tr, y[tr])
            ypath = model.predict_path_ls([X[te]] * len(Ks_tr))
        elif m == "kmp":
            model = KMP(rank=rank, delta=delta, lbd=0)
            model.fit(Ks_tr, y[tr])
            ypath = model.predict_path([X[te]] * len(Ks_tr))
        elif m == "icd":
            model = RidgeLowRank(method="icd",
                                 rank=rank, lbd=0)
            model.fit([Ksum_tr], y[tr])
            ypath = model.predict_path([X[te]])
        elif m == "nystrom":
            model = RidgeLowRank(method="nystrom",
                                 rank=rank, lbd=0)
            model.fit([Ksum_tr], y[tr])
            ypath = model.predict_path([X[te]])
        elif m == "csi":
            model = RidgeLowRank(method="csi", rank=rank,
                                 lbd=0, method_init_args={"delta": delta})
            model.fit([Ksum_tr], y[tr])
            ypath = model.predict_path([X[te]])
        elif m == "L2KRR":
            model = RidgeMKL(method="l2krr", lbd=0)
            model.fit(Ks=Ks, y=y, holdout=te)
            ypath = np.vstack([model.predict(te)] * rank).T
        else:
            raise ValueError(m)

        # Compute explained variance
        errs = np.zeros(ypath.shape[1])
        for j in range(ypath.shape[1]):
            errs[j] = (np.var(y[te]) - np.var(ypath[:, j] - y[te])) / np.var(y[te])
        results[m] = errs

    # Plot
    fname = os.path.join(out_dir, "test_mse_%s.pdf" % dataset)
    plt.figure()
    plt.title(dataset)
    for m in formats.keys():
        plt.plot(results[m], formats[m], label=m)
    plt.xlabel("Model capacity")
    plt.ylabel("Test MSE")
    plt.ylim((0, 1))
    plt.legend()
    plt.grid()
    plt.savefig(fname)
    plt.close()
    print("Written %s" % fname)


if __name__ == "__main__":
    for dset in KEEL_DATASETS:
        try:
            process(dset)
        except Exception as e:
            print("Exception with %s: %s" % (dset, e.message))

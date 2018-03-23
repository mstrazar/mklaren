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
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface

# Datasets
from datasets.keel import load_keel, KEEL_DATASETS

# Utils
import matplotlib.pyplot as plt

# New methods
from examples.lars.lars_mkl import LarsMKL
from examples.lars.lars_group import p_ri, p_const, p_sc, colors
from mklaren.mkl.kmp import KMP


# Parameters
out_dir = "/Users/martins/Dev/mklaren/examples/mkl/output"
N = 2000
delta = 5
gamma = .1
p_tr = .8
lbd = 0.001
rank = 30

models = ("lars-ri", "lars-co", "lars-sc", "kmp")
colors = {"lars-ri": "green", "lars-sc": "pink", "lars-co": "gray", "kmp": "red"}


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
    tr = np.random.choice(range(n), size=int(p_tr * n), replace=False)
    te = np.array(list(set(range(n)) - set(tr)))
    tr = tr[np.argsort(y[tr].ravel())]
    te = te[np.argsort(y[te].ravel())]

    # Training kernels
    Ks_tr = [Kinterface(data=X[tr],
                        kernel=exponential_kernel,
                        kernel_args={"gamma": gamma})
             for gamma in gamma_range]

    # Collect test error paths
    results = dict()
    for m in models:
        if m == "lars-ri":
            model = LarsMKL(delta=delta, rank=rank, f=p_ri)
            model.fit(Ks_tr, y[tr])
            ypath = model.predict_path_extend([X[te]] * len(Ks_tr))
        elif m == "lars-sc":
            model = LarsMKL(delta=delta, rank=rank, f=p_sc)
            model.fit(Ks_tr, y[tr])
            ypath = model.predict_path_extend([X[te]] * len(Ks_tr))
        elif m == "lars-co":
            model = LarsMKL(delta=delta, rank=rank, f=p_const)
            model.fit(Ks_tr, y[tr])
            ypath = model.predict_path_extend([X[te]] * len(Ks_tr))
        elif m == "kmp":
            model = KMP(rank=rank, delta=delta, lbd=0)
            model.fit(Ks_tr, y[tr])
            ypath = model.predict_path([X[te]] * len(Ks_tr))
        else:
            raise ValueError(m)
        errs = np.linalg.norm(ypath - y[te].reshape((len(te), 1)), axis=0)
        results[m] = errs

    # Plot
    fname = os.path.join(out_dir, "test_mse_%s.pdf" % dataset)
    plt.figure()
    plt.title(dataset)
    for m in models:
        plt.plot(results[m], label=m, color = colors[m])
    plt.xlabel("Model capacity")
    plt.ylabel("Test MSE")
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

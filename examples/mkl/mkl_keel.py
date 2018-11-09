hlp = """
    Trajectory of grpup LARS with different penalty functions 
    and KMP greedy algorithm.
"""

import scipy.stats as st
import numpy as np
import os
import csv
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
from examples.lars.lars_group import p_ri, p_const, p_sc, p_sig, p_act
from examples.mkl.mkl_est_var import estimate_sigma_dist


# Parameters
res_dir = "/Users/martins/Dev/mklaren/examples/mkl/results/mkl_keel"
out_dir = "/Users/martins/Dev/mklaren/examples/mkl/output/mkl_keel"
N = 2000
replicates = 10
no_kernels = 10
delta = 10
p_tr = .5
lbd = 0.000
rank = 100

formats = {"lars-ri": "gv-",
           # "lars-sc": "rv-",
           "lars-co": "cv-",
           "lars-sig": "bv-",
           "lars-act": "yv-",
           # "kmp": "c--",
           # "icd": "b--",
           # "nystrom": "m--",
           # "csi": "r--",
           "L2KRR": "k-"}

penalty = {
    "lars-ri": p_ri,
    "lars-sc": p_sc,
    "lars-co": p_const,
    "lars-sig": p_sig,
    "lars-act": p_act,
    }

header = ["replicate", "dataset", "method", "N", "rank", "delta", "evar"]


def process(dataset, repl=0):
    # Load data
    data = load_keel(n=N, name=dataset)

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

    # Estimate sigma range
    sigma_range = estimate_sigma_dist(X=X[tr], q=no_kernels)

    # Training kernels
    Ks_tr = [Kinterface(data=X[tr],
                        kernel=exponential_kernel,
                        kernel_args={"sigma": sigma})
             for sigma in sigma_range]
    Ks = [Kinterface(data=X,
                     kernel=exponential_kernel,
                     kernel_args={"sigma": sigma})
          for sigma in sigma_range]

    Ksum_tr = Kinterface(data=X[tr],
                         kernel=kernel_sum,
                         kernel_args={"kernels": [exponential_kernel] * len(sigma_range),
                                      "kernels_args": [{"sigma": sigma} for sigma in sigma_range]})

    # Collect test error paths
    results = dict()
    for m in formats.keys():
        if m.startswith("lars-"):
            model = LarsMKL(delta=delta, rank=rank, f=penalty[m])
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
    fname = os.path.join(out_dir, "test_mse_%s_%d.pdf" % (dataset, repl))
    plt.figure()
    plt.title(dataset)
    for m in sorted(formats.keys()):
        plt.plot(results[m], formats[m], label=m)
    plt.xlabel("Model capacity")
    plt.ylabel("Test MSE")
    plt.ylim((0, 1))
    plt.legend()
    plt.grid()
    plt.savefig(fname)
    plt.close()
    print("Written %s" % fname)

    # Generate row
    rows = list()
    for m, errs in results.items():
        rows.append({"replicate": repl,
                     "dataset": dataset,
                     "method": m,
                     "N": len(X),
                     "rank": rank,
                     "delta": delta,
                     "evar": np.mean(errs)})

    return rows


def main():
    # Seed
    np.random.seed(42)

    # Mkdir
    for dr in (out_dir, res_dir):
        if not os.path.exists(dr):
            os.makedirs(dr)
            print("Makedir %s" % dr)

    # Open output stream
    fname = os.path.join(res_dir, "results.csv")
    fp = open(fname, "w", buffering=0)
    writer = csv.DictWriter(fp, fieldnames=header)
    writer.writeheader()
    count = 0

    # Process
    for repl in range(replicates):
        for dset in KEEL_DATASETS:
            try:
                rows = process(dset, repl)
                if rows is not None:
                    count += 1
                    writer.writerows(rows)
                    print("Written %d rows to %s" % (count, fname))
            except Exception as e:
                print("Exception with %s: %s" % (dset, e.message))
    fp.close()
    print("End")


if __name__ == "__main__":
    main()

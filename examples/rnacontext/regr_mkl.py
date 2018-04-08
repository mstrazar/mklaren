hlp = """
    Trajectory of grpup LARS with different penalty functions 
    and KMP greedy algorithm.
"""

import itertools as it
import numpy as np
import csv
import os

os.environ["OCTAVE_EXECUTABLE"] = "/usr/local/bin/octave"

# Kernels
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface
from mklaren.regression.ridge import RidgeLowRank
from mklaren.mkl.kmp import KMP
from mklaren.regression.ridge import RidgeMKL

# New methods
from examples.lars.lars_mkl import LarsMKL
from examples.lars.lars_group import p_ri, p_const, p_sc
from scipy.stats import multivariate_normal as mvn

import scipy.stats as st
from mklaren.kernel.string_kernel import *
from mklaren.kernel.kinterface import Kinterface
from mklaren.regression.ridge import RidgeLowRank
from datasets.rnacontext import load_rna, RNA_OPTIMAL_K, dataset2spectrum, RNA_DATASETS
from examples.inducing_points.inducing_points import meth2color
from examples.strings.string_utils import generic_function_plot


import matplotlib.pyplot as plt

# Parameters
out_dir = "/Users/martins/Dev/mklaren/examples/rnacontext/output/regr_mkl"
N = 1000
delta = 5
p_tr = .8
lbd = 0.000
rank = 5
replicates = 30

formats = {"lars-ri": "gv-",
           # "lars-sc": "bv-",
           "lars-co": "cv-",
           "kmp": "c--",
           # "icd": "b--",
           # "nystrom": "m--",
           # "csi": "r--",
           "L2KRR": "k-"}

header = ["repl", "method", "N", "keff", "sigma", "noise", "snr", "evar", "ranking"]


def process(dataset=RNA_DATASETS[0], repl=0):
    """ Process one iteration of a dataset. """

    # Load data
    np.random.seed(42)
    k_range = range(2, 6)
    snr = 0
    rows = list()

    # Load data
    data = load_rna(dataset)
    inxs = np.argsort(st.zscore(data["target"]))
    X = data["data"][inxs]
    y = st.zscore(data["target"])[inxs]

    # Ground truth kernels
    Ks = [Kinterface(data=X,
                     kernel=string_kernel,
                     kernel_args={"mode": "1spectrum", "K": k})
          for k in k_range]

    # Training/test
    sample = np.random.choice(inxs, size=int(N), replace=False)
    tr, te = np.sort(sample[:int(N * p_tr)]), np.sort(sample[int(N * p_tr):])

    # Training kernels
    Ks_tr = [Kinterface(data=X[tr],
                        kernel=string_kernel,
                        kernel_args={"mode": "1spectrum", "K": k})
             for k in k_range]

    Ksum_tr = Kinterface(data=X[tr],
                         kernel=kernel_sum,
                         kernel_args={"kernels": [string_kernel] * len(k_range),
                                      "kernels_args": [{"K": k, "mode": "1spectrum"} for k in k_range]})

    # Collect test error paths
    results = dict()
    try:
        for m in formats.keys():
            if m == "lars-ri":
                model = LarsMKL(delta=delta, rank=rank, f=p_ri)
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
            evars = np.zeros(ypath.shape[1])
            for j in range(ypath.shape[1]):
                evars[j] = (np.var(y[te]) - np.var(ypath[:, j] - y[te])) / np.var(y[te])
            results[m] = evars
    except ValueError as ve:
        print("Exception", ve)
        return

    # Compute ranking
    # scores = dict([(m, np.mean(ev)) for m, ev in results.items()])
    scores = dict([(m, ev[-1]) for m, ev in results.items()])
    scale = np.array(sorted(scores.values(), reverse=True)).ravel()
    for m in results.keys():
        ranking = 1 + np.where(scale == scores[m])[0][0]
        row = {"repl": repl, "method": m, "N": N,
               "evar": scores[m], "ranking": ranking, "snr": snr}
        rows.append(row)

    return rows


if __name__ == "__main__":
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Makedir %s" % out_dir)

    # Write to .csv
    fname = os.path.join(out_dir, "results.csv")
    fp = open(fname, "w")
    writer = csv.DictWriter(fp, fieldnames=header)
    writer.writeheader()
    fp.close()

    # Process and write to file
    count = 0
    for repl in range(replicates):
        for dset in RNA_DATASETS:
            rows = process(dset, repl)
            writer.writerows(rows)
            count += len(rows)
            print("Written %s (%d)" % (fname, count))

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

# Datasets
from datasets.keel import load_keel, KEEL_DATASETS

# Utils
import matplotlib.pyplot as plt

# New methods
from examples.lars.lars_mkl import LarsMKL
from examples.lars.lars_group import p_ri, p_const, p_sc, colors
from scipy.stats import multivariate_normal as mvn

# Parameters
out_dir = "/Users/martins/Dev/mklaren/examples/mkl/results/mkl_sim"
N = 200
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


header = ["repl", "method", "N", "keff", "gamma", "noise", "evar", "ranking"]


def process():
    # Load data
    np.random.seed(42)
    noise_range = np.logspace(-5, 2, 8)
    gamma_range = np.logspace(-5, 5, 11)
    replicates = 30

    # Ground truth kernels
    X = np.linspace(-N, N, N).reshape((N, 1))
    Ks = [Kinterface(data=X,
                     kernel=exponential_kernel,
                     kernel_args={"gamma": gamma})
          for gamma in gamma_range]

    rows = list()
    count = 0
    for repl, noise in it.product(range(replicates), noise_range):

        # Simulate data from one kernel
        # keff = np.random.randint(0, len(Ks))
        keff = 2
        C = Ks[keff][:, :] + noise * np.eye(N)
        y = mvn.rvs(mean=np.zeros((N, )), cov=C)

        # Training/test
        tr = np.random.choice(range(N), size=int(p_tr * N), replace=False)
        te = np.array(list(set(range(N)) - set(tr)))
        tr = tr[np.argsort(y[tr].ravel())]
        te = te[np.argsort(y[te].ravel())]

        # Training kernels
        Ks_tr = [Kinterface(data=X[tr],
                            kernel=exponential_kernel,
                            kernel_args={"gamma": gamma})
                 for gamma in gamma_range]

        Ksum_tr = Kinterface(data=X[tr],
                          kernel=kernel_sum,
                          kernel_args={"kernels": [exponential_kernel] * len(gamma_range),
                                       "kernels_args": [{"gamma": gam} for gam in gamma_range]})

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
        except AssertionError:
            continue

        # Compute ranking
        scores = dict([(m, np.mean(ev)) for m, ev in results.items()])
        scale = np.array(sorted(scores.values(), reverse=True)).ravel()
        for m in results.keys():
            ranking = np.where(scale == scores[m])[0][0]
            row = {"repl": repl, "method": m, "noise": noise, "N": N,
                   "keff": keff, "gamma": gamma_range[keff],
                   "evar": scores[m], "ranking": ranking}
            rows.append(row)

        # Write to .csv
        fname = os.path.join(out_dir, "results.csv")
        fp = open(fname, "w")
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
        fp.close()
        print("Written %s (%d)" % (fname, count))
        count += 1


if __name__ == "__main__":
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Makedir %s" % out_dir)
    process()

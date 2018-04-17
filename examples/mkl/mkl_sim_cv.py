hlp = """
    Trajectory of grpup LARS with different penalty functions 
    and KMP greedy algorithm. Use cross validation.
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
from examples.lars.lars_group import p_ri, p_const, p_sc, p_sig, p_act
from examples.mkl.mkl_est_var import estimate_sigma_dist
from scipy.stats import multivariate_normal as mvn

# Parameters
out_dir = "/Users/martins/Dev/mklaren/examples/mkl/results/mkl_sim_cv"
delta = 5
p_tr = .6
p_va = .2
rank = 30

formats = {"lars-ri": "gv-",
           "lars-co": "cv-",
           "lars-sig": "bv-",
           # "lars-act": "yv-",
           # "lars-sc": "rv-",
           # "kmp": "c--",
           # "icd": "b--",
           # "nystrom": "m--",
           # "csi": "r--",
           # "L2KRR": "k-"
           }


header = ["repl", "method", "N", "d", "keff", "sigma", "noise", "snr", "evar", "ranking"]

penalty = {
    "lars-ri": p_ri,
    "lars-sc": p_sc,
    "lars-co": p_const,
    "lars-sig": p_sig,
    "lars-act": p_act,
    }


def process():
    # Load data
    np.random.seed(42)
    noise_range = np.logspace(-5, 2, 8)
    d_range = [1, 10, 100]
    N_range = [100, 300, 1000]
    replicates = 30

    rows = list()
    count = 0
    for repl, noise, d, N in it.product(range(replicates), noise_range, d_range, N_range):

        # Ground truth kernels
        X = np.random.randn(N, d)
        sigma_range = estimate_sigma_dist(X, 10)

        Ks = [Kinterface(data=X,
                         kernel=exponential_kernel,
                         kernel_args={"sigma": sigma})
              for sigma in sigma_range]

        # Simulate data from one kernel
        keff = 5
        f = mvn.rvs(mean=np.zeros((N,)), cov=Ks[keff][:, :])
        y = mvn.rvs(mean=f, cov=noise * np.eye(N))
        snr = np.round(np.var(f) / noise, 3)

        # Training/test; return a shuffled list
        sample = np.random.choice(range(N), size=N, replace=False)
        a, b = int(N * p_tr), int(N * (p_va + p_tr))
        tr, va, te = np.sort(sample[:a]), \
                     np.sort(sample[a:b]), \
                     np.sort(sample[b:])

        # Training kernels
        Ks_tr = [Kinterface(data=X[tr],
                            kernel=exponential_kernel,
                            kernel_args={"sigma": sigma})
                 for sigma in sigma_range]

        # Process
        results = dict()
        for m in formats.keys():
            model = LarsMKL(delta=delta, rank=rank, f=penalty[m])
            try:
                model.fit(Ks_tr, y[tr])
            except Exception as e:
                print("%s: %s" % (m, str(e)))
                continue

            y_va = y[va].reshape((len(va), 1))
            y_te = y[te].reshape((len(te), 1))

            ypath_va = model.predict_path_ls([X[va]] * len(Ks))
            ypath_te = model.predict_path_ls([X[te]] * len(Ks))

            scores_va = (np.var(y_va) - np.var(ypath_va - y_va, axis=0)) / np.var(y_va)
            scores_te = (np.var(y_te) - np.var(ypath_te - y_te, axis=0)) / np.var(y_te)

            t = np.argmax(scores_va)
            results[m] = np.round(scores_te[t], 3)

        # Compute ranking
        scores = dict([(m, ev) for m, ev in results.items()])
        scale = np.array(sorted(scores.values(), reverse=True)).ravel()
        for m in results.keys():
            ranking = 1 + np.where(scale == scores[m])[0][0]
            row = {"repl": repl, "method": m, "noise": noise, "N": N,
                   "keff": keff, "sigma": sigma_range[keff], "d": d,
                   "evar": scores[m], "ranking": ranking, "snr": snr}
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

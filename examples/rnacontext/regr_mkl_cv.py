hlp = """
    Trajectory of grpup LARS with different penalty functions 
    and KMP greedy algorithm.
"""
import numpy as np
import datetime
import pickle
import gzip
import csv
import os
import sys

os.environ["OCTAVE_EXECUTABLE"] = "/usr/local/bin/octave"

# New methods
from examples.lars.lars_mkl import LarsMKL
from examples.lars.lars_group import p_ri, p_const, p_sc, p_act, p_sig
import scipy.stats as st
from mklaren.kernel.kernel import linear_kernel
from mklaren.kernel.string_kernel import *
from mklaren.kernel.kinterface import Kinterface
from datasets.rnacontext import load_rna, RNA_OPTIMAL_K, dataset2spectrum, RNA_DATASETS

# Parameters
out_dir = "/Users/martins/Dev/mklaren/examples/rnacontext/results/regr_mkl_cv"
N = 3000
delta = 5
p_tr = .6
lbd = 0.000
rank = 100
replicates = 30

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

penalty = {
    "lars-ri": p_ri,
    "lars-sc": p_sc,
    "lars-co": p_const,
    "lars-sig": p_sig,
    "lars-act": p_act,
    }

header = ["repl", "dataset", "method", "N_tr", "N_va", "N_te", "evar", "ranking"]


def process(dataset=RNA_DATASETS[0], repl=0):
    """ Process one iteration of a dataset. """
    dat = datetime.datetime.now()
    print("\t%s\tdataset=%s cv=%d rank=%d" % (dat, dataset, repl, rank))

    # Load data
    np.random.seed(repl)
    K_range = range(3, 8)

    # Load data
    data = load_rna(dataset)
    inxs = np.argsort(st.zscore(data["target"]))
    y = st.zscore(data["target"])[inxs]

    # Training/test; return a shuffled list
    sample = np.random.choice(inxs, size=len(inxs), replace=False)
    a, b = int(N * p_tr), int(N)
    tr, va, te = np.sort(sample[:a]), \
                 np.sort(sample[a:b]), \
                 np.sort(sample[b:])

    # Load feature spaces
    try:
        Ys = [pickle.load(gzip.open(dataset2spectrum(dataset, K))) for K in K_range]
    except IOError:
        return None

    # Training kernels
    Ks_tr = [Kinterface(data=Y[tr],
                        kernel=linear_kernel,
                        row_normalize=True)
             for Y in Ys]

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

        ypath_va = model.predict_path_ls([Y[va] for Y in Ys])
        ypath_te = model.predict_path_ls([Y[te] for Y in Ys])

        scores_va = (np.var(y_va) - np.var(ypath_va - y_va, axis=0)) / np.var(y_va)
        scores_te = (np.var(y_te) - np.var(ypath_te - y_te, axis=0)) / np.var(y_te)

        t = np.argmax(scores_va)
        results[m] = np.round(scores_te[t], 3)

    # Compute ranking
    rows = list()
    scores = dict([(m, ev) for m, ev in results.items()])
    scale = np.array(sorted(scores.values(), reverse=True)).ravel()
    for m in results.keys():
        ranking = 1 + np.where(scale == scores[m])[0][0]
        row = {"dataset": dataset, "repl": repl, "method": m,
               "N_tr": len(tr), "N_va": len(te), "N_te": len(te),
               "evar": scores[m], "ranking": ranking}
        rows.append(row)

    return rows


if __name__ == "__main__":

    # Try to read out dir from input
    out_dir = sys.argv[1] if len(sys.argv) > 1 else out_dir
    print("Writing to %s" % out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Makedir %s" % out_dir)

    # Write to .csv
    fname = os.path.join(out_dir, "results.csv")
    fp = open(fname, "w", buffering=0)
    writer = csv.DictWriter(fp, fieldnames=header)
    writer.writeheader()

    # Process and write to file
    count = 0
    for repl in range(replicates):
        for dset in RNA_DATASETS:
            rows = process(dset, repl)
            if rows is not None:
                writer.writerows(rows)
                count += len(rows)
                print("Written %s (%d)" % (fname, count))

    fp.close()

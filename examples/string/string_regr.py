hlp = """
    Experiments with string regression on synthetic data.
"""

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

import os
import csv
import scipy.stats as st
import argparse

from scipy.linalg import sqrtm
from mklaren.kernel.string_kernel import *
from mklaren.kernel.string_util import *
from mklaren.kernel.kernel import kernel_sum
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from examples.string.string_utils import generic_function_plot
from examples.snr.snr import meth2color

# Hyperparameters
rank       = 3                                  # Approximation rank
delta      = 10                                 # Look-ahead hyperparameter
L          = 30                                 # Sequence length
trueK      = 4                                  # True substring length to generate hyperparameters
max_K      = 10                                 # Maximum substring (kmer) length
K_range    = range(1, max_K+1)                  # Range of string kernel hyperparameters
n_tr       = 500                                # Number of training data points
n_val      = 5000                               # Number of validation data points (hyperparameter selection)
n_te       = 5000                               # Number of test data points
cv_iter    = range(30)                          # Number of cross-validation iterations
lbd_range  = [0] + list(np.logspace(-5, 1, 7))  # Regularization parameter


def process(outdir):
    """
    Run experiments with epcified parameters.
    :param outdir: Output directory.
    :return:
    """

    # Fixed settings
    N = n_tr + n_val + n_te
    methods = ["Mklaren", "CSI", "Nystrom", "ICD"]

    # Fixed output
    # Create output directory
    if not os.path.exists(outdir): os.makedirs(outdir)
    fname = os.path.join(outdir, "results.csv")
    detname = os.path.join(outdir, "_details")
    if not os.path.exists(outdir): os.makedirs(outdir)
    if not os.path.exists(detname): os.makedirs(detname)
    print("Writing to %s ..." % fname)

    # Output
    header = ["n", "L", "iteration", "method", "lambda", "rank", "sp.corr", "sp.pval",
              "evar_tr", "evar_va", "evar", "mse"]
    fp = open(fname, "w", buffering=0)
    writer = csv.DictWriter(fp, fieldnames=header)
    writer.writeheader()

    # Training test split
    tr = range(0, n_tr)
    va = range(n_tr, n_tr+n_val)
    te = range(n_tr+n_val, n_tr+n_val+n_te)

    for cv in cv_iter:

        # Random subset of N sequences of length L
        X, _ = generate_data(N=N, L=L, p=0.0, motif="TGTG", mean=0, var=3, seed=cv)
        X = np.array(X)

        # Split into training in test set
        inxs = np.arange(N, dtype=int)
        np.random.shuffle(inxs)
        tr = inxs[tr]
        va = inxs[va]
        te = inxs[te]
        X_tr = X[tr]
        X_va = X[va]
        X_te = X[te]

        # Generate a sparse signal based on 4-mer composion (maximum lengthscale)
        act = np.random.choice(tr, size=rank, replace=False)
        K_full = Kinterface(data=X, kernel=string_kernel, kernel_args={"mode": SPECTRUM, "K": trueK},
                       row_normalize=False)
        K_act = K_full[:, act]
        H = K_act.dot(sqrtm(np.linalg.inv(K_act[act])))
        w = st.multivariate_normal.rvs(mean=np.zeros((rank,)), cov=np.eye(rank))
        y = H.dot(w)
        y_tr = y[tr]
        y_va = y[va]
        y_te = y[te]

        # Proposal kernels
        kargs = [{"mode": SPECTRUM, "K": k} for k in K_range]
        Ksum = Kinterface(data=X_tr, kernel=kernel_sum,
                              row_normalize=False,
                              kernel_args={"kernels": [string_kernel] * len(kargs),
                                           "kernels_args": kargs})
        Ks = [Kinterface(data=X_tr, kernel=string_kernel,
                         kernel_args=a, row_normalize=False) for a in kargs]

        # Modeling
        best_models =  {"True": {"y": y_te, "color": "black", "fmt": "--", }}

        for method in methods:
            best_models[method] = {"color": meth2color[method], "fmt": "-"}
            best_evar = -np.inf

            for lbd in lbd_range:
                if method == "Mklaren":
                    mkl = Mklaren(rank=rank, lbd=lbd, delta=delta)
                    try:
                        mkl.fit(Ks, y_tr)
                        yt = mkl.predict([X_tr] * len(Ks))
                        yv = mkl.predict([X_va] * len(Ks))
                        yp = mkl.predict([X_te] * len(Ks))
                    except Exception as e:
                        print(e)
                        continue
                else:
                    if method == "CSI":
                        model = RidgeLowRank(rank=rank, method="csi",
                                             method_init_args={"delta": delta}, lbd=lbd)
                    else:
                        model = RidgeLowRank(rank=rank, method=method.lower(), lbd=lbd)
                    try:
                        model.fit([Ksum], y_tr)
                        yt = model.predict([X_tr])
                        yv = model.predict([X_va])
                        yp = model.predict([X_te])
                    except Exception as e:
                        print(e)
                        continue

                # Store best *test* set prediction for a lambda
                spc = st.spearmanr(y_te, yp)
                evar_tr = (np.var(y_tr) - np.var(yt - y_tr)) / np.var(y_tr)
                evar_va = (np.var(y_va) - np.var(yv - y_va)) / np.var(y_va)
                evar = (np.var(y_te) - np.var(yp - y_te)) / np.var(y_te)
                mse = np.var(yp - y_te)
                if evar_va > best_evar:
                    best_evar = evar_va
                    best_yp = yp
                    best_models[method]["y"] = best_yp
                    print("Best lambda for %s: %.3E, expl. var.: %.3f" % (method, lbd, float(evar_va)))

                # Store row for each methods
                row = {"n": N, "L": L, "method": method, "rank": rank, "iteration": cv,
                         "sp.corr": spc[0], "sp.pval": spc[1], "lambda": lbd,
                       "evar_tr": evar_tr, "evar_va": evar_va, "evar": evar, "mse": mse}
                writer.writerow(row)

        # Plot a generic function plot for all methods, selecting best lambda
        fname = os.path.join(detname, "cv_K-%d_cv-%d.pdf" % (trueK, cv))
        generic_function_plot(f_out=fname, Ks=Ks, X=X_te,
                              models=best_models,
                              xlabel="K-mer length",
                              xnames=K_range,
                              truePar=K_range.index(trueK),
                              seed=0)


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description=hlp)
    parser.add_argument("output",  help="Output directory.")
    args = parser.parse_args()

    # Output directory
    out_dir = args.output
    process(out_dir)
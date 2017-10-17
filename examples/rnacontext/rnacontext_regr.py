hlp = """
    Experiments on RNAcontext string regression datasets. The methods Mklaren, CSI, Nystrom, ICD
    are run with string kernels to predict the RNA-binding protein binding affinity towards a given
    RNA sequence.
"""


import matplotlib
matplotlib.use("Agg")

import os
import csv
import time
import datetime
import argparse
import pickle, gzip
import scipy.stats as st
from mklaren.kernel.string_kernel import *
from mklaren.mkl.mklaren import Mklaren
from mklaren.kernel.kinterface import Kinterface
from mklaren.regression.ridge import RidgeLowRank
from datasets.rnacontext import load_rna, RNA_OPTIMAL_K, dataset2spectrum, RNA_DATASETS
from examples.inducing_points.inducing_points import meth2color
from examples.strings.string_utils import generic_function_plot


# Hyperparameters
rnk         = 5                                  # Approximation rank
lbd_range   = [0] + list(np.logspace(-5, 1, 7))  # Regularization parameter
iterations  = range(30)                          # Cross-validation iterations
delta       = 10                                 # Look-ahead parameter
n_tr        = 3000                               # Size of the training set of sequences
n_val       = 3000                               # Size of the validation set of sequences (fitting hyperparameters)


def process(dataset, outdir):
    """
    Run experiments with epcified parameters.
    :param dataset: Dataset key.
    :param outdir: Output directory.
    :return:
    """

    # List available kernels
    K_range = range(1, 11)
    kargs = [{"mode": SPECTRUM, "K": kl} for kl in K_range]
    kernels = ",".join(set(map(lambda t: t["mode"], kargs)))

    # Fixed settings
    methods = ["Mklaren", "CSI", "Nystrom", "ICD"]
    rank_range = (rnk,)
    trueK = RNA_OPTIMAL_K.get(dataset, None)

    # Fixed output
    # Create output directory
    detname = os.path.join(outdir, "_%s" % dataset)
    if not os.path.exists(outdir): os.makedirs(outdir)
    if not os.path.exists(detname): os.makedirs(detname)
    fname = os.path.join(outdir, "%s.csv" % dataset)
    print("Writing to %s ..." % fname)

    # Output
    header = ["dataset", "n", "L", "kernels", "method", "rank", "iteration", "lambda",
              "pivots", "time", "evar_tr", "evar_va", "evar", "mse"]
    fp = open(fname, "w", buffering=0)
    writer = csv.DictWriter(fp, fieldnames=header, quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writeheader()

    # Load data
    data = load_rna(dataset)
    X = data["data"]
    y = st.zscore(data["target"])
    n, L = len(X), len(X[0])

    # Load feature spaces
    Ys = [pickle.load(gzip.open(dataset2spectrum(dataset, K))) for K in K_range]

    # Generate random datasets and perform prediction
    seed = 0
    for cv in iterations:

        # Select random test/train indices
        np.random.seed(seed)
        inxs = np.arange(n, dtype=int)
        np.random.shuffle(inxs)
        tr = inxs[:n_tr]
        va = inxs[n_tr:n_tr + n_val]
        te = inxs[n_tr + n_val:]

        # Training / test split
        y_tr = y[tr]
        y_va = y[va]
        y_te = y[te]

        # Print after dataset generation
        dat = datetime.datetime.now()
        print("%s\tdataset=%s cv=%d (computing kernels...)" % (dat, dataset, cv))

        # For plotting
        X_te = X[te]
        Ks = [Kinterface(kernel=string_kernel, data=X[tr], kernel_args=arg) for arg in kargs]

        # Precomputed kernel matrices
        Ls_tr = [np.array(Y[tr, :].dot(Y[tr, :].T).todense()) for Y in Ys]
        Ls_va = [np.array(Y[va, :].dot(Y[tr, :].T).todense()) for Y in Ys]
        Ls_te = [np.array(Y[te, :].dot(Y[tr, :].T).todense()) for Y in Ys]
        Ls_tr_sum = [sum(Ls_tr)]
        Ls_va_sum = [sum(Ls_va)]
        Ls_te_sum = [sum(Ls_te)]

        # Modeling
        for rank in rank_range:
            dat = datetime.datetime.now()
            print("\t%s\tdataset=%s cv=%d rank=%d" % (dat, dataset, cv, rank))
            best_models = {"True": {"y": y_te, "color": "black", "fmt": "--", }}
            for method in methods:
                best_models[method] = {"color": meth2color[method], "fmt": "-"}
                best_evar = -np.inf

                for lbd in lbd_range:
                    t1 = time.time()
                    if method == "Mklaren":
                        mkl = Mklaren(rank=rank, lbd=lbd, delta=delta)
                        try:
                            mkl.fit(Ls_tr, y_tr)
                            yt = mkl.predict(Xs=None, Ks=Ls_tr)
                            yv = mkl.predict(Xs=None, Ks=Ls_va)
                            yp = mkl.predict(Xs=None, Ks=Ls_te)
                            pivots = ",".join(map(lambda pi: str(K_range[pi]), mkl.G_mask.astype(int)))
                        except Exception as e:
                            print(e)
                            continue
                    else:
                        pivots = ""
                        if method == "CSI":
                            model = RidgeLowRank(rank=rank, method="csi",
                                                 method_init_args={"delta": delta}, lbd=lbd)
                        else:
                            model = RidgeLowRank(rank=rank, method=method.lower(), lbd=lbd)
                        try:
                            model.fit(Ls_tr_sum, y_tr)
                            yt = model.predict(Xs=None, Ks=Ls_tr_sum)
                            yv = model.predict(Xs=None, Ks=Ls_va_sum)
                            yp = model.predict(Xs=None, Ks=Ls_te_sum)
                        except Exception as e:
                            print(e)
                            continue
                    t2 = time.time() - t1

                    # Evaluate explained variance on the three sets
                    evar_tr = (np.var(y_tr) - np.var(yt - y_tr)) / np.var(y_tr)
                    evar_va = (np.var(y_va) - np.var(yv - y_va)) / np.var(y_va)
                    evar    = (np.var(y_te) - np.var(yp - y_te)) / np.var(y_te)
                    mse     = np.var(yp - y_te)

                    # Select best lambda to plot
                    if evar_va > best_evar:
                        best_evar = evar_va
                        best_yp = yp
                        best_models[method]["y"] = best_yp

                    # Write to output
                    row = {"L": L, "n": len(X), "method": method, "dataset": dataset,
                           "kernels": kernels, "rank": rank, "iteration": cv, "lambda": lbd,
                           "time": t2, "evar_tr": evar_tr, "evar_va": evar_va, "evar": evar,
                           "mse": mse, "pivots": pivots}

                    writer.writerow(row)
                    seed += 1

            # Plot a function fit after selecting best lambda
            fname = os.path.join(detname, "%s.generic_plot_cv-%d_rank-%d.pdf" % (dataset, cv, rank))
            generic_function_plot(f_out=fname, Ks=Ks, X=X_te,
                                  models=best_models,
                                  xlabel="K-mer length",
                                  xnames=K_range,
                                  truePar=K_range.index(trueK) if trueK else None)


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description=hlp)
    parser.add_argument("dataset", help="Dataset. One of {%s}." % ", ".join(RNA_DATASETS))
    parser.add_argument("output",  help="Output directory.")
    args = parser.parse_args()

    # Output directory
    data_set = args.dataset
    out_dir = args.output
    assert data_set in RNA_DATASETS
    process(data_set, out_dir)
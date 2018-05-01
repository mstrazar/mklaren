hlp = """
    Test prediction with low-rank approximations on the energy dataset with matern, exponential and
    periodic kernels. Tested methods are Mklaren, ICD, CSI, Nystrom, FITC, RFF_KMP.

"""

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

import argparse
import csv
import os
import numpy as np
import itertools as it
from mklaren.kernel.kernel import exponential_kernel, kernel_sum, matern32_gpy, periodic_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.regression.spgp import SPGP
from sklearn.metrics import mean_squared_error as mse
from datasets.energy import load_energy
from examples.inducing_points.inducing_points import plot_signal_subplots, test

# Hyperparameters
delta        = 10                                   # Look-ahead parameter
rank_range   = (7, )                                # Tested rank range
lambda_range = [0] + list(np.logspace(-1, 1, 5))    # L2-regularization parameter range


def process(dataset, kernel, outdir):
    """
    Run experiments with epcified parameters.
    :param dataset: Dataset key.
    :param kernel: Kernel function name.
    :param outdir: Output directory.
    :return:
    """

    kernel_function = None
    pars = dict()
    methods = []

    # Experiment parameters
    if kernel == "exponential":
        kernel_function = exponential_kernel
        pars = {"gamma": np.logspace(-4, 4, 5)}
        methods = ("Mklaren", "ICD", "CSI", "Nystrom", "SPGP", "RFF", "RFF-NS")

    elif kernel == "matern":
        kernel_function = matern32_gpy
        pars = {"lengthscale": SPGP.gamma2lengthscale(np.logspace(-4, 4, 5))}
        methods = ("Mklaren", "ICD", "CSI", "Nystrom", "SPGP")

    elif kernel == "periodic":
        kernel_function = periodic_kernel
        pars = {"sigma": np.logspace(-2, 2, 5),
                "per": np.logspace(1, 3, 10)}
        methods = ("Mklaren", "ICD", "CSI", "Nystrom", )


    # Data parameters
    signals = ["T%d" % i for i in range(1, 10)]
    assert dataset in signals
    inxs = range(1, 19)

    # Store results
    if not os.path.exists(outdir): os.makedirs(outdir)
    fname = os.path.join(outdir, "%s.csv" % dataset)
    subdname = os.path.join(outdir, "_%s" % dataset)
    if not os.path.exists(subdname):
        os.makedirs(subdname)

    header = ["experiment", "signal",  "tsi", "method",
              "mse_val", "mse_f", "mse_y", "rank", "lbd", "n", "time"]
    writer = csv.DictWriter(open(fname, "w", buffering=0), fieldnames=header)
    writer.writeheader()

    # Load energy consumption data
    data = load_energy(signal=dataset)
    data_out = load_energy(signal="T_out")
    X, labels = data["data"], data["labels"]
    X_out, _ = data_out["data"], data_out["labels"]

    # Normalize by outside temperature
    # Assume true signal is the mean of the signals
    Y = X - X_out
    N, n = Y.shape

    # Training, validation, test
    x  = np.atleast_2d(np.arange(0, n, 3)).T
    xv = np.atleast_2d(np.arange(1, n, 3)).T
    xp = np.atleast_2d(np.arange(2, n, 3)).T

    Nt, Nv, Np = x.shape[0], xv.shape[0], xp.shape[0]
    f = Y[1:19, x].mean(axis=0).ravel()

    for rank, lbd, tsi in it.product(rank_range, lambda_range, inxs):
        y = Y[tsi, x].reshape((Nt, 1))
        yv = Y[tsi, xv].reshape((Nv, 1))
        yp = Y[tsi, xp].reshape((Np, 1))

        # Sum and List of kernels
        vals = list(it.product(*pars.values()))
        names = pars.keys()
        Klist = [Kinterface(data=x, kernel=kernel_function, row_normalize=True,
                            kernel_args=dict([(n, v) for n, v in zip(names, vlist)])) for vlist in vals]
        Ksum = Kinterface(data=x, kernel=kernel_sum, row_normalize=True,
                            kernel_args={"kernels": [kernel_function] * len(vals),
                                         "kernels_args": [dict([(n, v) for n, v in zip(names, vlist)])
                                                          for vlist in vals]})
        # Fit models and plot signal
        # Remove True anchors, as they are non-existent
        try:
            models_val = test(Ksum=Ksum, Klist=Klist,
                          inxs=range(rank),
                          X=x, Xp=xv, y=y, f=f,  delta=delta, lbd=lbd,
                          methods=methods)
            models = test(Ksum=Ksum, Klist=Klist,
                          inxs=range(rank),
                          X=x, Xp=xp, y=y, f=f, delta=delta, lbd=lbd,
                          methods=methods)
        except Exception as e:
            print("Exception: %s %s" % (e, e.message))
            continue
        del models["True"]

        # Store file as pdf + eps
        fname = os.path.join(subdname, "plot_multi-%s_tsi-%d_lbd-%.3f_rank-%d.pdf" % (dataset, tsi, lbd, rank))
        plot_signal_subplots(X=xp, Xp=xp, y=yp, f=None, models=models, f_out=fname)

        fname = os.path.join(subdname, "plot_multi-%s_tsi-%d_lbd-%.3f_rank-%d.eps" % (dataset, tsi, lbd, rank))
        plot_signal_subplots(X=xp, Xp=xp, y=yp, f=None, models=models, f_out=fname)

        for ky in models.keys():
            mse_yv = mse(models_val[ky]["yp"].ravel(), yv.ravel())
            mse_yp = mse(models[ky]["yp"].ravel(), yp.ravel())

            time = models[ky]["time"]
            row = {"experiment": kernel,
                   "signal": dataset, "tsi": tsi, "method": ky,
                   "mse_f": 0, "mse_y": mse_yp, "mse_val": mse_yv,
                   "rank": rank, "lbd": lbd, "n": n, "time": time}
            writer.writerow(row)


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description=hlp)
    parser.add_argument("dataset", help="Time series. One of {T1-T9}.")
    parser.add_argument("kernel",  help="Kernel function used for modelling. One of {exponential, periodic, matern}.")
    parser.add_argument("output",  help="Output directory.")
    args = parser.parse_args()

    # Output directory
    data_set = args.dataset
    kern = args.kernel
    out_dir = args.output
    process(data_set, kern, out_dir)
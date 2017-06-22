if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

import csv
import datetime
import os

import sys
import numpy as np
import itertools as it
from mklaren.kernel.kernel import exponential_kernel, kernel_sum, matern32_gpy, periodic_kernel
from mklaren.kernel.kinterface import Kinterface
from sklearn.metrics import mean_squared_error as mse

from datasets.energy import load_energy
from examples.snr.snr import plot_signal, test


# Dataset description
# https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction#
args = dict(enumerate(sys.argv))
ename = args.get(1, "exponential")
print("Proceeding with %s scenario ..." % ename)

kernel_function = None
pars = dict()
methods = []

# Experiment parameters
if ename == "exponential":
    kernel_function = exponential_kernel
    pars = {"gamma": np.logspace(-4, 4, 5),}
    methods = ("Mklaren", "ICD", "CSI", "Nystrom", "FITC", "RFF")

elif ename == "matern":
    kernel_function = matern32_gpy
    pars = {"lengthscale": np.logspace(-4, 4, 5)}
    methods = ("Mklaren", "ICD", "CSI", "Nystrom", "FITC")

elif ename == "periodic":
    kernel_function = periodic_kernel
    pars = {"sigma": np.logspace(-2, 2, 5),
            "per": np.logspace(1, 3, 10)}
    methods = ("Mklaren", "ICD", "CSI", "Nystrom", )


# Hyperparameters
delta = 10
rank_range = (7, 14, 21)
lambda_range = [0] + list(np.logspace(-1, 1, 5))

# Data parameters
signals = ["T%d" % i for i in range(1, 10)]
inxs = range(1, 19)

# Store results
d = datetime.datetime.now()
dname = os.path.join("..", "output", "energy", "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname): os.makedirs(dname)
rcnt = len(os.listdir(dname))
fname = os.path.join(dname, "results_%d.csv" % rcnt)
subdname = os.path.join(dname, "details_%d" % rcnt)
if not os.path.exists(subdname):
    os.makedirs(subdname)

header = ["experiment", "signal",  "tsi", "method",
          "mse_val", "mse_f", "mse_y", "rank", "lbd", "n", "time"]
writer = csv.DictWriter(open(fname, "w", buffering=0), fieldnames=header)
writer.writeheader()

for sig in signals:

    # Load energy condumption data
    data = load_energy(signal=sig)
    data_out = load_energy(signal="T_out")
    X, labels = data["data"], data["labels"]
    X_out, _ = data_out["data"], data_out["labels"]

    # Normalize by outside temperature
    # Assume true signal is the mean of the signals
    Y = X - X_out
    N, n = Y.shape

    # Training, validation, test
    if ename == "periodic":
        x = np.atleast_2d(np.arange(0, n/2)).T
        xv = np.atleast_2d(np.arange(n/2, n, 2)).T
        xp = np.atleast_2d(np.arange(n/2+1, n, 2)).T
    else:
        x  = np.atleast_2d(np.arange(0, n, 3)).T
        xv = np.atleast_2d(np.arange(1, n, 3)).T
        xp = np.atleast_2d(np.arange(2, n, 3)).T

    Nt, Nv, Np = x.shape[0], xv.shape[0], xp.shape[0]
    f = Y[1:19, x].mean(axis=0).ravel()

    for rank, lbd in it.product(rank_range, lambda_range):
        for tsi in inxs:
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
            except:
                continue
            del models["True"]

            # Store file
            fname = os.path.join(subdname, "plot_sig-%s_tsi-%d_lbd-%.3f_rank-%d.pdf" % (sig, tsi, lbd, rank))
            plot_signal(X=xp, Xp=xp, y=yp, f=yp, models=models, f_out=fname,
                        typ="plot_models")

            for ky in models.keys():
                mse_yv = mse(models_val[ky]["yp"].ravel(), yv.ravel())
                mse_yp = mse(models[ky]["yp"].ravel(), yp.ravel())
                mse_f = mse(models[ky]["yp"].ravel(), fp.ravel())
                time = models[ky]["time"]
                row = {"experiment": ename,
                       "signal": sig,  "tsi": tsi, "method": ky,
                       "mse_f": mse_f, "mse_y": mse_yp, "mse_val": mse_yv,
                       "rank": rank, "lbd": lbd, "n": n, "time": time}
                writer.writerow(row)
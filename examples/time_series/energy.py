if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

import csv
import datetime
import os

import numpy as np
import itertools as it
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface
from sklearn.metrics import mean_squared_error as mse

from datasets.energy import load_energy
from examples.snr.snr import plot_signal, test


# Dataset description
# https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction#

# Hyperparameters
gamma_range = np.logspace(-4, 4, 5)
delta = 10
rank_range = (7, 14, 21)
lambda_range = [0] + list(np.logspace(-1, 1, 5))

# Data parameters
signals = ["T%d" % i for i in range(1, 10)]
inxs = range(1, 19)
methods=("Mklaren", "ICD", "CSI", "Nystrom", "FITC", "RFF")

# Store results
d = datetime.datetime.now()
dname = os.path.join("..", "output", "energy", "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname): os.makedirs(dname)
rcnt = len(os.listdir(dname))
fname = os.path.join(dname, "results_%d.csv" % rcnt)
subdname = os.path.join(dname, "details_%d" % rcnt)
if not os.path.exists(subdname):
    os.makedirs(subdname)

header = ["signal",  "tsi", "method", "mse_val", "mse_f", "mse_y", "rank", "lbd", "n"]
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
    x  = np.atleast_2d(np.arange(0, n, 3)).T
    xv = np.atleast_2d(np.arange(1, n, 3)).T
    xp = np.atleast_2d(np.arange(2, n, 3)).T
    Nt, Nv, Np = x.shape[0], xv.shape[0], xp.shape[0]
    fv = Y[1:19, xv].mean(axis=0).ravel()
    fp = Y[1:19, xp].mean(axis=0).ravel()

    for rank, lbd in it.product(rank_range, lambda_range):
        for tsi in inxs:
            y = Y[tsi, x].reshape((Nt, 1))
            yv = Y[tsi, xv].reshape((Nv, 1))
            yp = Y[tsi, xp].reshape((Np, 1))

            # Sum and List of kernels
            Klist = [Kinterface(data=x, kernel=exponential_kernel,
                                kernel_args={"gamma": g}) for g in gamma_range]
            Ksum = Kinterface(data=x, kernel=kernel_sum,
                                kernel_args={"kernels": [exponential_kernel] * len(gamma_range),
                                             "kernels_args": [{"gamma": g} for g in gamma_range]})

            # Fit models and plot signal
            # Remove True anchors, as they are non-existent
            try:
                models_val = test(Ksum=Ksum, Klist=Klist,
                              inxs=range(rank),
                              X=x, Xp=xv, y=y, f=fv,  delta=delta, lbd=lbd,
                              methods=methods)
                models = test(Ksum=Ksum, Klist=Klist,
                              inxs=range(rank),
                              X=x, Xp=xp, y=y, f=fp, delta=delta, lbd=lbd,
                              methods=methods)
            except:
                continue
            del models["True"]

            # Store file
            fname = os.path.join(subdname, "plot_sig-%s_tsi-%d_lbd-%.3f_rank-%d.pdf" % (sig, tsi, lbd, rank))
            plot_signal(X=x, Xp=x, y=y, f=fp, models=models, f_out=fname,
                        typ="plot_models")

            for ky in models.keys():
                mse_yv = mse(models_val[ky]["yp"].ravel(), yv.ravel())
                mse_yp = mse(models[ky]["yp"].ravel(), yp.ravel())
                mse_f = mse(models[ky]["yp"].ravel(), fp.ravel())
                row = {"signal": sig,  "tsi": tsi, "method": ky,
                       "mse_f": mse_f, "mse_y": mse_yp, "mse_val": mse_yv,
                       "rank": rank, "lbd": lbd, "n": n}
                writer.writerow(row)
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

import csv
import datetime
import os

import numpy as np
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
rank = 7
lbd = 0

# Data parameters
signals = ["T%d" % i for i in range(1, 10)]
inxs = range(1, 19)
methods=("Mklaren", "ICD", "CSI", "Nystrom", "FITC")

# Store results
d = datetime.datetime.now()
dname = os.path.join("..", "output", "energy", "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname): os.makedirs(dname)
rcnt = len(os.listdir(dname))
fname = os.path.join(dname, "results_%d.csv" % rcnt)
subdname = os.path.join(dname, "details_%d" % rcnt)
if not os.path.exists(subdname):
    os.makedirs(subdname)

header = ["signal",  "tsi", "method", "mse_f", "mse_y", "rank", "lbd", "n"]
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
    x = np.arange(n).reshape((n, 1))
    f = Y[1:19, :n].mean(axis=0)

    for tsi in inxs:
        y = Y[tsi, :n].reshape((n, 1))

        # Sum and List of kernels
        Klist = [Kinterface(data=x, kernel=exponential_kernel, kernel_args={"gamma": g}) for g in gamma_range]
        Ksum = Kinterface(data=x, kernel=kernel_sum,
                            kernel_args={"kernels": [exponential_kernel] * len(gamma_range),
                                         "kernels_args": [{"gamma": g} for g in gamma_range]})

        # Fit models and plot signal
        # Remove True anchors, as they are non-existent
        try:
            models = test(Ksum=Ksum, Klist=Klist,
                          inxs=list(np.linspace(0, n-1, 7, dtype=int)),
                          X=x, Xp=x, y=y, f=f,  delta=delta, lbd=lbd,
                          methods=methods)
        except:
            continue
        del models["True"]

        # Store file
        fname = os.path.join(subdname, "plot_sig-%s_tsi-%d.pdf" % (sig, tsi))
        plot_signal(X=x, Xp=x, y=y, f=f, models=models, f_out=fname,
                    typ="plot_models")

        for ky in models.keys():
            mse_yp = mse(models[ky]["yp"].ravel(), y.ravel())
            mse_f = mse(models[ky]["yp"].ravel(), f.ravel())

            row = {"signal": sig,  "tsi": tsi, "method": ky,
                   "mse_f": mse_f, "mse_y": mse_yp,
                   "rank": rank, "lbd": lbd, "n": n}
            writer.writerow(row)
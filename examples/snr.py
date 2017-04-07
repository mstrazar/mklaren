"""
Motivation: If one is allowed to sample columns with regularization,
this leads to lower errors at high noise levels.

Comparison of the three pivot selection methods with varying noise rates
on a simple Gaussian Process signal.

"""
import matplotlib
matplotlib.use("Agg")

import csv
import datetime
import os
import itertools as it

import numpy as np
from scipy.stats import multivariate_normal as mvn
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt



# Create output directory
d = datetime.datetime.now()
dname = os.path.join("output", "snr", "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname):
    os.makedirs(dname)
fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
print("Writing to %s ..." % fname)

header = ["repl", "method", "n", "lbd", "snr", "rank", "noise", "mse_sig"]
writer = csv.DictWriter(open(fname, "w", buffering=0),
                        fieldnames=header, quoting=csv.QUOTE_ALL)
writer.writeheader()

# Parameters
n_range = [30, 100, 300, 1000]
noise_range = np.logspace(-2, 2, 10)
repeats = range(10)
gamma = 1
rank = 4
methods = ["Mklaren", "CSI", "ICD"]
count = 0
lbd_range = [0, 1, 3]


for repl, n, noise, lbd in it.product(repeats, n_range, noise_range, lbd_range):

    # Generate data
    X = np.linspace(-10, 10, n).reshape((n, 1))
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 1})

    f = mvn.rvs(mean=np.zeros((n, )), cov=K[:, :])
    y = mvn.rvs(mean=f, cov=noise * np.eye(n, n))

    # Mklaren
    yp = None
    rows = []
    for method in methods:
        if method == "Mklaren":
            mkl = Mklaren(rank=rank,
                          delta=10, lbd=lbd)
            mkl.fit([K], y)
            yp = mkl.predict([X])

        elif method == "ICD":
            icd = RidgeLowRank(rank=rank, lbd=lbd,
                               method="icd")
            icd.fit([K], y)
            yp = icd.predict([X])

        elif method == "CSI":
            icd = RidgeLowRank(rank=rank, lbd=lbd,
                               method="csi", method_init_args={"delta": 10})
            icd.fit([K], y)
            yp = icd.predict([X])


        mse_sig = mse(yp, f)
        snr = np.var(f) / noise

        row = {"repl": repl, "method": method, "n": n, "snr": snr, "lbd": lbd,
               "rank": rank, "noise": np.round(np.log10(noise), 2), "mse_sig": mse_sig}
        rows.append(row)

    if len(rows) == len(methods):
        writer.writerows(rows)
        count += len(rows)
        print("%s Written %d rows (n=%d)" % (str(datetime.datetime.now()), count, n))





# Plot archive snippet
# Plot data
# plt.figure()
# plt.plot(X.ravel(), f, "--", color="gray")
# plt.plot(X.ravel(), y, ".", color="black")
# plt.plot(X.ravel(), y_mkl, "-", color="green", linewidth=2)
# plt.plot(X.ravel(), y_icd, "-", color="blue", linewidth=2)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
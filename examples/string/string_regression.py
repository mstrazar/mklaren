import os
import csv
import numpy as np
import scipy.stats as st
import datetime
import itertools as it
import matplotlib.pyplot as plt
from mklaren.kernel.string_kernel import *
from mklaren.kernel.string_util import *
from mklaren.kernel.kernel import kernel_sum
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank

# List available kernels
args = [
    {"mode": SPECTRUM, "K": 2},
    {"mode": SPECTRUM, "K": 3},
    {"mode": SPECTRUM, "K": 4},
    {"mode": SPECTRUM, "K": 5},
    {"mode": WD, "K": 2},
    {"mode": WD, "K": 4},
    {"mode": WD, "K": 5},
    {"mode": WD_PI, "K": 2},
    {"mode": WD_PI, "K": 3},
    {"mode": WD_PI, "K": 4},
]

# Hyperparameters
methods = ["CSI", "Mklaren"]
# p_range = range(1, len(args)+1)
p_range = [len(args)]
seed_range = range(10)
rank = 5
delta = 10
lbd = 0.01
var = 10

# Fixed output
# Create output directory
d = datetime.datetime.now()
dname = os.path.join("..", "output", "string",
                     "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname): os.makedirs(dname)
rcnt = len(os.listdir(dname))
fname = os.path.join(dname, "results_%d.csv" % rcnt)
print("Writing to %s ..." % fname)

# Output
header = ["n", "method", "rank", "iteration", "lambda",
          "p", "corr", "corr.p"]
fp = open(fname, "w", buffering=0)
writer = csv.DictWriter(fp, fieldnames=header)
writer.writeheader()

# Training test split
tr = range(0, 50)
te = range(50, 100)

# Generate random datasets and perform prediction
count = 0
for num_k, seed in it.product(p_range, seed_range):
    print("Seed: %d, num. kernels: %d" % (seed, num_k))
    X, y = generate_data(N=100, L=100, p=0.5, motif="TGTG", mean=0, var=var, seed=seed)
    Xa = np.array(X)
    X_tr = Xa[tr]
    X_te = Xa[te]
    y_tr = y[tr]
    y_te = y[te]

    # Individual kernels
    Ks = [Kinterface(kernel=string_kernel, data=X_tr, kernel_args=arg)
          for arg in args[:num_k]]

    # Sum of kernels
    Ksum = Kinterface(data=X_tr, kernel=kernel_sum,
                      row_normalize=True,
                      kernel_args={"kernels": [string_kernel] * len(args[:num_k]),
                                   "kernels_args": args[:num_k]})

    # Modeling
    for method in methods:
        if method == "CSI":
            csi = RidgeLowRank(rank=rank, method="csi",
                               method_init_args={"delta": delta}, lbd=lbd)
            try:
                csi.fit([Ksum], y_tr)
                yp = csi.predict([X_te])
            except Exception as e:
                print(e)
                continue

        elif method == "Mklaren":
            mkl = Mklaren(rank=rank, lbd=lbd, delta=delta)
            try:
                mkl.fit(Ks, y_tr)
                yp = mkl.predict([X_te] * len(Ks))
            except Exception as e:
                print(e)
                continue

        sp_c, sp_p = st.spearmanr(yp, y_te)

        row = {"n": len(X), "method": method,
               "rank": rank, "iteration": seed, "lambda": lbd,
                "p": num_k, "corr": sp_c, "corr.p": sp_p}
        writer.writerow(row)
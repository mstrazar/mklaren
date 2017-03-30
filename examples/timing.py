from mklaren.kernel.kernel import poly_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from time import time
from datetime import datetime
from itertools import product
import numpy as np
import csv
import os

# Fixed hyper parameters
repeats = 10
range_n = np.linspace(1e3, 1e6, 4)
range_degree = [6]
range_repeat = range(repeats)
range_lbd = [0.01, 0.1, 1, 10]
range_rank = [30]
sigma2 = 0.1    # noise variance
lbd = 0.1

methods = ["Mklaren"]
delta = 10  # Delta to max rank
P = 1   # Number of true kernels to be taken in the sum
p_tr = 0.6
p_va = 0.2
p_te = 0.2



# Create output directory
d = datetime.now()
dname = os.path.join("output", "timing", "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname):
    os.makedirs(dname)
fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
print("Writing to %s ..." % fname)

header = ["repl", "method",  "n", "D", "rank", "time"]


writer = csv.DictWriter(open(fname, "w", buffering=0),
                        fieldnames=header, quoting=csv.QUOTE_ALL)
writer.writeheader()

count = 0
for repl, n, maxd, rank in product(range_repeat, range_n, range_degree, range_rank):
    print("%s Fitting with parameters: %s" % (str(datetime.now()), str([repl, n, maxd, rank])))

    # Rank is known because the data is simulated
    X = np.random.rand(n, range_rank[-1])
    X = (X - X.mean(axis=0)) / np.std(X, axis=0)
    y_true = np.random.rand(n, 1)
    y_true = y_true - y_true.mean()

    Ks_all = []
    for d in range(0, maxd + 1):
        K_a = Kinterface(kernel=poly_kernel, kernel_args={"degree": d},
                         data=X, row_normalize=False)
        Ks_all.append(K_a)
    rows = []
    for method in methods:
        valid = True
        try:
            t1 = time()
            if method == "Mklaren":
                model = Mklaren(rank=rank, delta=delta, lbd=lbd)
                model.fit(Ks_all, y_true)
            elif method == "CSI":
                model = RidgeLowRank(rank=rank, method="csi", lbd=lbd,
                                         method_init_args={"delta": delta},
                                         sum_kernels=True)
                model.fit(Ks_all, y_true)
            elif method == "ICD":
                model = RidgeLowRank(rank=rank, method="icd", lbd=lbd,
                                         sum_kernels=True)
                model.fit(Ks_all, y_true)
            t2 = time() - t1
        except:
            print("%s error" % method)
            continue

        # Score the predictions
        if valid:
            row = {"repl": repl, "method": method, "time": t2,
               "n": n, "D": maxd, "rank": rank}
            rows.append(row)

    if len(rows) == len(methods):
        count += len(rows)
        writer.writerows(rows)
        print("Written %d rows" % count)
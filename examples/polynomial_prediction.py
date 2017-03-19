from mklaren.kernel.kernel import poly_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from sklearn.metrics import mean_squared_error
from itertools import product
import numpy as np
import csv
import datetime
import os

# Fixed hyper parameters
repeats = 10
range_n = [30, 100, 300, 1000]
range_degree = range(2, 7)
range_repeat = range(repeats)
range_lbd = [0, 1, 10]
range_rank = [3, 5, 10]

methods = ["Mklaren", "CSI", "ICD"]
delta = 10  # Delta to max rank
p_tr = 0.75
p_te = 1.0 - p_tr
P = 1   # Number of true kernels to be taken in the sum


# Create output directory
d = datetime.datetime.now()
dname = os.path.join("output", "polynomial_prediction", "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname):
    os.makedirs(dname)
fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
print("Writing to %s ..." % fname)

header = ["repl", "method", "mse_fit", "mse_pred", "lbd", "n", "D", "norm", "rank"]
writer = csv.DictWriter(open(fname, "w", buffering=0),
                        fieldnames=header, quoting=csv.QUOTE_ALL)
writer.writeheader()

count = 0
for repl, n, lbd, maxd, rank in product(range_repeat, range_n, range_lbd, range_degree, range_rank):

    # Training / test split
    tr = range(int(n * p_tr))
    te = range(tr[-1]+1, n)

    # Rank is known because the data is simulated
    X = np.random.rand(n, range_rank[-1])
    X = (X - X.mean(axis=0)) / np.std(X, axis=0)

    X_tr = X[tr]
    X_te = X[te]

    Ks_tr = []
    Ks_all = []
    for d in range(1, maxd + 1):
        K_tr = Kinterface(kernel=poly_kernel, kernel_args={"degree": d},
                          data=X_tr, row_normalize=True)
        K_a = Kinterface(kernel=poly_kernel, kernel_args={"degree": d},
                         data=X, row_normalize=True)
        Ks_tr.append(K_tr)
        Ks_all.append(K_a)

    mu_true = np.zeros((len(Ks_tr),))
    mu_true[-P] = 1

    # True kernel matrix to generate the signal ;
    # weights are defined only by the training set
    alpha = np.random.rand(n, 1)
    alpha[te] = 0
    K_true = sum([mu_true[i] * Ks_all[i][:, :] for i in range(len(Ks_all))])
    y_true = K_true.dot(alpha)
    norm = np.linalg.norm(K_true)

    rows = []
    for method in methods:
        # Fit the mklaren method and predict
        try:
            if method == "Mklaren":
                model_mklaren = Mklaren(rank=rank, delta=delta, lbd=lbd)
                model_mklaren.fit(Ks_tr, y_true[tr])
                y_pred = model_mklaren.predict([X_te] * len(Ks_tr))
                y_fit = model_mklaren.predict([X_tr] * len(Ks_tr))
                w_fit = model_mklaren.mu / model_mklaren.mu.sum()
            elif method == "CSI":
                model_csi = RidgeLowRank(rank=rank, method="csi", lbd=lbd,
                                         method_init_args={"delta": delta},
                                         sum_kernels=True)
                model_csi.fit(Ks_tr, y_true[tr])
                y_pred = model_csi.predict([X_te] * len(Ks_tr))
                y_fit = model_csi.predict([X_tr] * len(Ks_tr))
            elif method == "ICD":
                model_icd = RidgeLowRank(rank=rank, method="icd", lbd=lbd,
                                         sum_kernels=True)
                model_icd.fit(Ks_tr, y_true[tr])
                y_pred = model_icd.predict([X_te] * len(Ks_tr))
                y_fit = model_icd.predict([X_tr] * len(Ks_tr))
        except:
            print("%s error" % method)
            continue

        # Score the predictions
        mse_pred = mean_squared_error(y_true[te], y_pred)
        mse_fit = mean_squared_error(y_true[tr], y_fit)
        row = {"repl": repl, "method": method, "mse_fit": mse_fit, "mse_pred": mse_pred,
               "lbd": lbd, "n": n, "D": maxd, "norm": norm, "rank": rank}
        rows.append(row)

    if len(rows) == len(methods):
        count += len(rows)
        writer.writerows(rows)
        print("Written %d rows" % count)
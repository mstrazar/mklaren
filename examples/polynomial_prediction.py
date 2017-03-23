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
P = 1   # Number of true kernels to be taken in the sum
p_tr = 0.6
p_va = 0.2
p_te = 0.2

# Create output directory
d = datetime.datetime.now()
dname = os.path.join("output", "polynomial_prediction", "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname):
    os.makedirs(dname)
fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
print("Writing to %s ..." % fname)

header = ["repl", "method", "mse_fit", "mse_pred", "expl_var_fit",
          "expl_var_pred", "lbd", "n", "D", "norm", "rank"]
writer = csv.DictWriter(open(fname, "w", buffering=0),
                        fieldnames=header, quoting=csv.QUOTE_ALL)
writer.writeheader()

count = 0
for repl, n, maxd, rank in product(range_repeat, range_n, range_degree, range_rank):

    # Training / validation / test split
    tr = range(int(n * p_tr))
    va = range(tr[-1]+1, int(n * (p_tr + p_va)))
    te = range(va[-1]+1, n)

    # Rank is known because the data is simulated
    X = np.random.rand(n, range_rank[-1])
    X = (X - X.mean(axis=0)) / np.std(X, axis=0)

    X_tr = X[tr]
    X_va = X[va]
    X_te = X[te]

    Ks_tr = []
    Ks_all = []
    for d in range(0, maxd + 1):
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
        # Select lambda via cross-validation; store results as you go.
        mse_best = float("inf")
        lbd_best = 0
        y_pred = y_fit = mse_pred = mse_fit = expl_var_pred = None

        for lbd in range_lbd:
            try:
                model = None
                if method == "Mklaren":
                    model = Mklaren(rank=rank, delta=delta, lbd=lbd)
                    model.fit(Ks_tr, y_true[tr])
                    w_fit = model.mu / model.mu.sum()
                elif method == "CSI":
                    model = RidgeLowRank(rank=rank, method="csi", lbd=lbd,
                                             method_init_args={"delta": delta},
                                             sum_kernels=True)
                    model.fit(Ks_tr, y_true[tr])
                elif method == "ICD":
                    model = RidgeLowRank(rank=rank, method="icd", lbd=lbd,
                                             sum_kernels=True)
                    model.fit(Ks_tr, y_true[tr])

                y_val = model.predict([X_va] * len(Ks_tr))
                mse_val = mean_squared_error(y_true[va], y_val)
                if mse_val < mse_best:
                    mse_best, lbd_best = mse_val, lbd

                    # Fit MSE
                    y_fit = model.predict([X_tr] * len(Ks_tr))
                    mse_fit = mean_squared_error(y_true[tr], y_fit)
                    total_mse_fit = mean_squared_error(y_true[tr], np.zeros((len(tr),)))
                    expl_var_fit = (total_mse_fit - mse_fit) / total_mse_fit

                    # Predict MSE
                    y_pred = model.predict([X_te] * len(Ks_tr))
                    mse_pred = mean_squared_error(y_true[te], y_pred)
                    total_mse_pred = mean_squared_error(y_true[te], np.zeros((len(te),)))
                    expl_var_pred = (total_mse_pred - mse_pred) / total_mse_pred
            except:
                print("%s error" % method)
                continue

        # Score the predictions
        if y_pred is not None:
            row = {"repl": repl, "method": method, "mse_fit": mse_fit, "mse_pred": mse_pred,
                   "expl_var_fit": expl_var_fit, "expl_var_pred": expl_var_pred,
                   "lbd": lbd_best, "n": n, "D": maxd, "norm": norm,
                   "rank": rank}
            rows.append(row)

    if len(rows) == len(methods):
        count += len(rows)
        writer.writerows(rows)
        print("Written %d rows" % count)
"""

No regularization. Isolation of pivot selection via LAR criterion and CSI.

"""
import numpy as np
import itertools as it
import csv
import os
import datetime

from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from mklaren.kernel.kernel import linear_kernel
from mklaren.kernel.kinterface import Kinterface

from examples.features import Features
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

# Create output directory
d = datetime.datetime.now()
dname = os.path.join("output", "polynomial_overfit", "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname):
    os.makedirs(dname)
fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
print("Writing to %s ..." % fname)

header = ["repl", "method", "rank", "D", "mse_fit", "mse_pred", "expl_var_pred"]
writer = csv.DictWriter(open(fname, "w", buffering=0),
                        fieldnames=header, quoting=csv.QUOTE_ALL)
writer.writeheader()



# Full polynomial kernels
n = 300
p = 10
degree_range = range(1, 6) + ["inf"]
rank_range = [10, 30, 50]
repeats = range(10)
methods = ["Mklaren", "CSI"]

# Training/test split
p_tr = 0.7
tr = range(int(p_tr * n))
te = range(int(p_tr * n), n)

count = 0
for repl, rank, d in it.product(repeats, rank_range, degree_range):

    # Generate random data and center
    np.random.seed(repl)
    X = np.random.randn(n, p)
    # beta = np.random.randn(p).reshape((p, 1))
    # y = X.dot(beta)
    # y = y - y.mean()

    # Create high dimensional features (& remove bias)
    feats = Features(degree=d)
    F = feats.fit_transform(X)[:, 1:]

    # Normalize feats
    F = (F - F.mean(axis=0)) / np.std(F, axis=0)
    K_tr = Kinterface(data=F[tr],
                      kernel=linear_kernel)

    # Signal is function of the training set
    alpha = np.random.randn(n).reshape((n, 1))
    alpha[te] = 0
    y = linear_kernel(F, F).dot(alpha).ravel()
    y = y - y.mean()

    rows = []
    mse_fit = mse_pred = None
    for method in methods:

        # Fit Mklaren
        if method == "Mklaren":
            mklaren = Mklaren(rank=rank, delta=p, lbd=0)
            try:
                mklaren.fit([K_tr], y[tr])
                yp_tr = mklaren.predict([F[tr]]).ravel()
                yp_te = mklaren.predict([F[te]]).ravel()
            except Exception as e:
                print("Mklaren error: %s" % str(e))
                continue

        # Fit CSI
        elif method == "CSI":
            csi = RidgeLowRank(rank=rank, lbd=0,
                               method="csi",
                               method_init_args={"delta": p})
            try:
                csi.fit([K_tr], y[tr])
                yp_tr = csi.predict([F[tr]]).ravel()
                yp_te = csi.predict([F[te]]).ravel()
            except Exception as e:
                print("CSI error: %s" % str(e))
                continue

        # Linear regression baseline
        elif method == "LinReg":
            model = LinearRegression()
            model.fit(F[tr], y[tr])
            yp_tr = model.predict(F[tr])
            yp_te = model.predict(F[te])
        elif method == "KRR":
            model = KernelRidge(kernel="linear", alpha=0)
            model.fit(F[tr], y[tr])
            yp_tr = model.predict(F[tr])
            yp_te = model.predict(F[te])

        mse_fit = mse(yp_tr, y[tr])
        mse_pred = mse(yp_te, y[te])
        var_y = np.var(y[te])
        expl_var_pred = (var_y - mse_pred) / var_y

        rows.append({"repl": repl,
                     "method": method,
                     "rank": rank,
                     "D": d,
                     "mse_fit": mse_fit,
                     "mse_pred": mse_pred,
                     "expl_var_pred": expl_var_pred})

    if len(rows) == len(methods):
        writer.writerows(rows)
        count += len(rows)
        print("Written %d rows." % count)

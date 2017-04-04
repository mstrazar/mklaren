import numpy as np
import itertools as it
import datetime
import os
import csv

from datasets.blitzer import load_books, load_dvd,\
    load_electronics, load_kitchen
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank

from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import linear_kernel, poly_kernel

from sklearn.metrics import mean_squared_error as mse
from sklearn.kernel_ridge import KernelRidge

# Experiment paramters
delta = 10
p_train = 0.7
repeats = 10
methods = ["CSI", "Mklaren"]
max_features = 4000
n_range = [None]
rank_range = [10, 20, 40]
kappa_range = np.linspace(0, 1, 4)
lbd_range = [0] + list(np.logspace(-3, 3, 10))
degree_range = [1, ]

# Datasets and options
datasets = {
    "books":        load_books,
    "kitchen":      load_kitchen,
    "dvd":          load_dvd,
    "electronics":  load_electronics,
}

# Create output directory
d = datetime.datetime.now()
dname = os.path.join("output", "blitzer_low_rank", "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname):
    os.makedirs(dname)
fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
print("Writing to %s ..." % fname)

# Prepare CSV
count = 0
header = ["dataset", "repl", "method", "n", "degree", "rank", "kappa", "lbd", "rmse_tr", "rmse_pred"]
writer = csv.DictWriter(open(fname, "w", buffering=0),
                        fieldnames=header, quoting=csv.QUOTE_ALL)
writer.writeheader()
for repl in range(repeats):
    for datatag, load in datasets.iteritems():
        for n, rank, degree in it.product(n_range, rank_range, degree_range):
            data = load(n=n, max_features=max_features)

            # Split data in training and test
            inxs = range(len(data["target"]))
            np.random.seed(repl)
            np.random.shuffle(inxs)
            tr, va = inxs[:int(p_train * len(inxs))], \
                     inxs[int(p_train * len(inxs)):]

            # Training data
            X_tr = data["data"][tr].toarray()
            y_tr = data["target"][tr]
            bias = np.mean(y_tr)
            y_tr = y_tr - bias

            X_va = data["data"][va].toarray()
            y_va = data["target"][va] - bias

            # Idpt. test data
            X_te = data["data_test"].toarray()
            y_te = data["target_test"] - bias

            K_tr = Kinterface(kernel=poly_kernel, data=X_tr, kernel_args={"degree": degree})

            rows = list()
            for method in methods:
                d = datetime.datetime.now()
                print("%s Fitting %s, rank: %d, shape (tr): %s, shape(te): %s"
                      % (str(d), method, rank, str(X_tr.shape), str(X_te.shape)) )
                mse_va_best = float("inf")
                lbd_best, kappa_best = None, None
                if method == "KRR":
                    for lbd in lbd_range:
                        model = KernelRidge(alpha=lbd, kernel="linear")
                        model.fit(X_tr, y_tr)
                        yp_va = model.predict(X_va)
                        mse_va = mse(y_va, yp_va)
                        if mse_va < mse_va_best:
                            mse_va_best = mse_va
                            yp_tr = model.predict(X_tr)
                            yp_te = model.predict(X_te)
                            G = X_tr
                            lbd_best = lbd

                elif method == "Mklaren":
                    for lbd in lbd_range:
                        model = Mklaren(rank=rank, delta=delta, lbd=lbd)
                        try:
                            model.fit([K_tr], y_tr)
                        except Exception, e:
                            print("%s error: %s" % (method, e))
                            continue
                        yp_va = model.predict([X_va])
                        mse_va = mse(y_va, yp_va)
                        if mse_va < mse_va_best:
                            mse_va_best = mse_va
                            yp_tr = model.predict([X_tr])
                            yp_te = model.predict([X_te])
                            G = model.G
                            lbd_best = lbd

                elif method == "CSI":
                    for lbd, kappa in it.product(lbd_range, kappa_range):
                        model = RidgeLowRank(rank=rank,
                                             method="csi",
                                             method_init_args = {"delta": delta, "kappa": kappa}, lbd=lbd)
                        try:
                            model.fit([K_tr], y_tr)
                        except Exception, e:
                            print("%s error: %s" % (method, e))
                            continue
                        yp_va = model.predict([X_va])
                        mse_va = mse(y_va, yp_va)
                        if mse_va < mse_va_best:
                            mse_va_best = mse_va
                            yp_tr = model.predict([X_tr])
                            yp_te = model.predict([X_te])
                            G = sum(model.Gs)
                            lbd_best, kappa_best = lbd, kappa

                # Compute reconstruction error & MSE
                if mse_va_best < float("inf"):
                    # recon = np.linalg.norm(K_tr[:, :] - G.dot(G.T), ord="fro") / (X_tr.shape[0] * X_tr.shape[1])
                    rmse_tr = mse(yp_tr, y_tr)**0.5
                    rmse_pred = mse(yp_te, y_te)**0.5

                    rows.append({"dataset": datatag, "method": method,
                                 "n": len(X_tr) + len(X_va), "rank": rank, "kappa": kappa_best,
                                 "lbd": lbd_best,
                                 "degree": degree,
                                 # "recon": recon,
                                 "rmse_tr": rmse_tr,
                                 "rmse_pred": rmse_pred,
                                 "repl": repl})



            if len(rows) == len(methods):
                writer.writerows(rows)
                count += len(rows)
                print("Written %d rows" % count)
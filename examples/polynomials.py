from mklaren.mkl.mklaren import Mklaren
from mklaren.projection.csi import CSI
from mklaren.projection.icd import ICD
from itertools import product
from mklaren.kernel.kernel import kernel_row_normalize, poly_kernel
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime
import os
import csv

# Create output directory
d = datetime.datetime.now()
dname = os.path.join("output", "polynomials", "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname):
    os.makedirs(dname)
fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
print("Writing to %s ..." % fname)

header = ["exp.id", "method", "rank", "D", "pr", "pr_pval",
               "sp", "sp_pval", "sp_w", "sp_w_pval", "mse"]
writer = csv.DictWriter(open(fname, "w", buffering=0),
                        fieldnames=header, quoting=csv.QUOTE_ALL)
writer.writeheader()


# Fixed settings
n = 100             # Number of examples
P = 2               # Last p true kernels
repeats = 30        # Sampling repeats
delta = 10

# Range settings
range_degrees = range(2, 10)
range_ranks = [3, 10, 30,]

# Training and test sets
tr = range(int(n * 0.75))
te = range(n - int(n * 0.25), n)

count = 0
for repl in range(repeats):
    for rank, D in product(range_ranks, range_degrees):
        # Random parametric model from weights of the training set
        rows = list()
        count += 1
        alpha = np.random.rand(n, 1).ravel()

        # Start with an arbitrarily big, random dataset
        X = np.random.rand(n, rank)
        X = (X - X.mean(axis=0)) / np.std(X, axis=0)

        # Generate all kernels of up to a maximum a degree
        Ks = []
        for d in range(D):
            K = kernel_row_normalize(X.dot(X.T)**d)
            Ks.append(K)

        # Generate target signal from weights obtained on the training set only
        K_true = sum(Ks[-P:])
        K_all = sum(Ks)
        y = K_true.dot(alpha)

        # True weights
        mu_true = np.zeros((len(Ks), ))
        mu_true[-P:] = 1

        # Mklaren
        try:
            model = Mklaren(rank=rank, delta=delta )
            model.fit(Ks, y)
        except:
            print("Mklaren error")
            continue
        y_pred = model.y_pred.ravel()

        rho, pv = spearmanr(mu_true, model.mu)
        rho_fit, pv_fit = spearmanr(y_pred.ravel(), y.ravel())
        p_rho_fit, p_pv_fit = pearsonr(y_pred.ravel(), y.ravel())
        mse = np.linalg.norm(y_pred - y)

        row = {"exp.id": count, "method": "Mklaren", "rank": rank, "D": D, "pr": p_rho_fit, "pr_pval": p_pv_fit,
               "sp": rho_fit, "sp_pval": pv_fit, "sp_w": rho, "sp_w_pval": pv,
                "mse": mse}
        rows.append(row)

        # CSI
        try:
            csi = CSI(rank=rank, delta=delta, kappa=0.99)
            csi.fit(K_all, y)
            lin_model = LinearRegression()
            lin_model.fit(csi.G, y)
        except:
            print("CSI error")
            continue

        y_pred = lin_model.predict(csi.G).ravel()
        rho_fit, pv_fit = spearmanr(y_pred, y)
        p_rho_fit, p_pv_fit = pearsonr(y_pred.ravel(), y.ravel())
        mse = np.linalg.norm(y_pred - y)

        row = {"exp.id": count, "method": "CSI", "rank": rank, "D": D, "pr": p_rho_fit, "pr_pval": p_pv_fit,
               "sp": rho_fit, "sp_pval": pv_fit, "mse": mse}
        rows.append(row)

        writer.writerows(rows)
        print("Written %d rows" % count)


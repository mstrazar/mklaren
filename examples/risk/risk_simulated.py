hlp = """
    Risk vs. test error on simulated datasets with known ground truth kernel.
"""
import numpy as np
import os
import csv
import itertools as it
os.environ["OCTAVE_EXECUTABLE"] = "/usr/local/bin/octave"

# Kernels
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.regression.ridge import RidgeLowRank

# Utils
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

# New methods
from examples.lars.lars_kernel import lars_kernel, lars_map_Q
from examples.risk.risk_keel import fit_gamma

# Parameters
out_dir = "/Users/martins/Dev/mklaren/examples/risk/output/risk_simulated/"
fname = os.path.join(out_dir, "results.csv")

# Experiment parameters
delta_range = (2, 5, 10)
replicates = range(30)
n_range = (100, 300, 1000)

# Fixed
rank = 30
noise = 0.03
gamma = 0.0003
p_tr = .8
lbd = 0.00

# Models
models = ("lars", "icd", "csi")
colors = {"lars": "orange",
          "lars_ls": "red", "icd": "blue", "lars_no": "yellow", "KRR": "black", "csi": "magenta"}

# Results to save
results = list()

# Iterate
for n, repl in it.product(n_range, replicates):

    # Simulate data
    X = np.linspace(-1000, 1000, n).reshape((n, 1))
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": gamma})
    y = mvn.rvs(mean=np.zeros(n,), cov=K[:, :] + noise * np.eye(n))

    # Fit model
    gfit = fit_gamma(X, y)
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": gfit})

    # Training/test split
    tr = np.random.choice(range(n), size=int(p_tr * n), replace=False)
    te = np.array(list(set(range(n)) - set(tr)))
    tr = sorted(tr)
    te = sorted(te)
    K_tr = Kinterface(data=X[tr], kernel=exponential_kernel, kernel_args={"gamma": gfit})

    for d in delta_range:
        # Process
        epath = dict()
        ypath = dict()
        for m in models:
            # Fit path for each model
            try:
                if m == "lars":
                    Q, R, path, mu, act = lars_kernel(K_tr, y[tr], rank=rank, delta=d)
                    Qt = lars_map_Q(X[te], K_tr, Q, R, act)
                    yp = Qt.dot(path.T)
                elif m == "icd":
                    icd = RidgeLowRank(lbd=lbd, rank=rank, method="icd")
                    icd.fit([K_tr], y[tr])
                    icd.path_compute([X[tr]], y[tr])
                    yp = icd.path_predict([X[te]])
                elif m == "csi":
                    csi = RidgeLowRank(lbd=lbd, rank=rank, method="csi", method_init_args={"delta": d})
                    csi.fit([K_tr], y[tr])
                    csi.path_compute([X[tr]], y[tr])
                    yp = csi.path_predict([X[te]])
                else:
                    raise ValueError(m)
            except Exception as e:
                print("Exception: %s (%s)" % (str(e), m))
                continue

            # Compute explained variance for everything on the test path
            evar = np.zeros((yp.shape[1]), )
            for j in range(yp.shape[1]):
                evar[j] = (np.var(y[te]) - np.var(y[te] - yp[:, j])) / np.var(y[te])
            epath[m] = evar
            ypath[m] = yp

        if len(epath) != len(models):
            print("Invalidating replicate %d (%d/%d)" % (repl, len(epath), len(models)))
            continue

        # Measure rate of progress
        scores = dict([(m, np.round(np.mean(epath[m]), 3)) for m in models])
        for m in models:
            ecs = scores[m]
            ranking = sorted(scores.values(), reverse=True).index(ecs) + 1
            row = {"iter": repl, "model": m.upper(), "n": n, "delta": d, "rank": rank,
                   "evar": ecs, "ranking": ranking}
            results.append(row)

# Write to csv
fp = open(fname, "w")
writer = csv.DictWriter(fp, fieldnames=results[0].keys())
writer.writeheader()
writer.writerows(results)
fp.close()
print("Written %s" % fname)


if False:
    # Plot data and fit
    plt.figure()
    plt.plot(X.ravel(), y, ".")
    for m in models:
        plt.plot(X[te], ypath[m][:, -1], color=colors[m], label=m)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()

    # Test explained variance
    plt.figure()
    for m in models:
        plt.plot(epath[m], label="%s (%.2f)" % (m, max(epath[m])), color=colors[m])
    plt.xlabel("Model capacity $\\rightarrow$")
    plt.ylabel("Test Expl. var")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.show()


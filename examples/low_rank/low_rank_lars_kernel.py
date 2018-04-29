hlp = """
    Evaluation of low-rank kernel approximation methods
    with ridge regression on standard datasets from KEEL.
"""

import os
import csv
import sys
import itertools as it
import scipy.stats as st
import time
import argparse
import numpy as np

# Low-rank approximation methods
from mklaren.regression.ridge import RidgeLowRank
from mklaren.regression.ridge import RidgeMKL
from mklaren.regression.fitc import FITC
from mklaren.projection.rff import RFF_KMP
from mklaren.mkl.mklaren import Mklaren

# Kernels
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface
from time import time

# Datasets
from datasets.keel import load_keel, KEEL_DATASETS

# Utils
from numpy import var, mean, logspace
from random import shuffle, seed
from math import ceil
import matplotlib.pyplot as plt



# New methods
from examples.lars.lars_kernel import lars_kernel, lars_kernel_predict
from examples.lars.risk import estimate_risk, estimate_sigma
from examples.lars.lars_beta import plot_path, plot_residuals

# Comparable parameters with effective methods
N = 3000                            # Max number of data points
rank_range = range(2, 100)                   # Rang of tested ranks
p_range = (10,)                           # Fixed number of kernels
lbd_range = [0] + list(logspace(-5, 1, 7))  # Regularization parameter
delta = 10                              # Number of look-ahead columns (CSI and mklaren)
cv_iter = 5                               # Cross-validation iterations
training_size = 0.6                             # Training set fraction
validation_size = 0.2                             # Validation set (for fitting hyperparameters)
test_sizes = 0.2                             # Test set (for reporting scores)


# Method classes and fixed hyperparameters
methods = {
    "CSI" :        (RidgeLowRank, {"method": "csi", "method_init_args": {"delta": delta}}),
    "ICD" :        (RidgeLowRank, {"method": "icd"}),
    "Nystrom":     (RidgeLowRank, {"method": "nystrom"}),
    "CSI*" :       (RidgeLowRank, {"method": "csi", "method_init_args": {"delta": delta}}),
    "ICD*" :       (RidgeLowRank, {"method": "icd"}),
    "Nystrom*":    (RidgeLowRank, {"method": "nystrom"}),
    "Mklaren":     (Mklaren,      {"delta": delta}),
    "RFF_KMP":     (RFF_KMP, {"delta": delta}),
    "SPGP":        (FITC,         {}),
    "uniform":     (RidgeMKL,     {"method": "uniform"}),
    "L2KRR":       (RidgeMKL,     {"method": "l2krr"}),
}


def process():
    """
    Run experiments with specified parameters.
    :param dataset: Dataset key.
    :param outdir: Output directory.
    :return:
    """

    # Load data
    dataset = "quake"
    data = load_keel(n=N, name=dataset)

    # Parameters
    rank = 100
    delta = 10
    lbd = 1
    gamma = 10.0

    # Load data and normalize columns
    X = st.zscore(data["data"], axis=0)
    y = st.zscore(data["target"])
    inxs = np.argsort(y)
    X = X[inxs, :]
    y = y[inxs]
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": gamma})

    # Fit library models
    # model = RidgeLowRank(lbd=lbd, rank=rank, method="csi", method_init_args={"delta": delta})
    t1 = time()
    model = RidgeLowRank(lbd=lbd, rank=rank, method="icd")
    model.fit([K], y)
    yp = model.predict([X])
    t1 = time() - t1
    print("ICD time: %f" % t1)

    # Fit Kernel-LARS
    t1 = time()
    Q, R, path, mu, act = lars_kernel(K, y, rank=rank, delta=delta)
    t1 = time() - t1
    print("LARS time: %f" % t1)

    # Compute risk
    _, sigma_est = estimate_sigma(K[:, :], y)
    Cp_est = np.zeros(path.shape[0])
    for i, b in enumerate(path):
        mu = Q.dot(b)
        Cp_est[i] = estimate_risk(Q[:, :i+1], y, mu, sigma_est)

    # Plot fit
    plt.figure()
    plt.plot(y, ".")
    plt.plot(yp, "--", label="ICD")
    plt.plot(mu, "-", label="LARS")
    plt.legend()
    plt.show()

    # Diagnostic LARS plots
    plot_residuals(Q, y, path, tit="LARS")
    plot_path(path, tit="LARS")

    # Risk estimation
    plt.figure()
    plt.plot(Cp_est)
    plt.xlabel("Model capacity $\\rightarrow$")
    plt.ylabel("$C_p$")
    plt.grid()


if __name__ == "__main__":
    process()
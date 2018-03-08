hlp = """
    Risk vs. test error on standard datasets.
"""
import scipy.stats as st
import numpy as np
import os


# Kernels
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.regression.ridge import RidgeLowRank
from mklaren.regression.ridge import RidgeMKL

# Datasets
from datasets.keel import load_keel, KEEL_DATASETS

# Utils
import matplotlib.pyplot as plt

# New methods
from examples.lars.lars_kernel import lars_kernel, lars_kernel_predict, lars_map_Q
from examples.lars.risk import estimate_risk, estimate_sigma


# Parameters
out_dir = "/Users/martins/Dev/mklaren/examples/risk/output"
N = 3000
N = 1000
delta = 0
gamma = .1
p_tr = .2
lbd = 0.001

models = ("lars", "lars_ls", "icd", "KRR")
colors = {"lars": "orange", "lars_ls": "red", "icd": "blue", "lars_no": "yellow", "KRR": "black"}


def process(dataset):
    # Load data
    data = load_keel(n=N, name=dataset)

    # Evaluate ranks
    Rmax = min(int(p_tr * len(data["data"]))-12, 50)
    rank = Rmax
    rank_range = range(2, Rmax)
    print "Processing dataset %s with max rank %d" % (dataset, Rmax)

    # Load data and normalize columns
    X = st.zscore(data["data"], axis=0)
    y = st.zscore(data["target"])
    inxs = np.argsort(y)
    X = X[inxs, :]
    y = y[inxs]

    # Training/test
    n = len(X)
    tr = np.random.choice(range(n), size=int(p_tr * n), replace=False)
    te = np.array(list(set(range(n)) - set(tr)))
    tr = tr[np.argsort(y[tr])]
    te = te[np.argsort(y[te])]

    # Fit models
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": gamma})
    K_tr = Kinterface(data=X[tr], kernel=exponential_kernel, kernel_args={"gamma": gamma})
    Q, R, path, mu, act = lars_kernel(K_tr, y[tr], rank=rank, delta=delta)
    Qt = lars_map_Q(X=X[te], K=K_tr, act=act, Q=Q, R=R)

    # Estimate risk
    Cp = np.zeros((len(rank_range)))
    mse = np.zeros((len(rank_range)))
    _, sigma_est = estimate_sigma(K_tr[:, :], y[tr])
    evar = dict([(m, np.zeros((len(rank_range)))) for m in models])

    # LARS only
    for i, r in enumerate(rank_range):
        mu = Q.dot(path[r]).ravel()
        yp = lars_kernel_predict(X=X[te], K=K_tr, act=act, Q=Q, R=R, beta=path[r])
        Cp[i] = estimate_risk(Q[:, :r], y[tr], mu, sigma_est)
        mse[i] = np.var(y[te] - yp).ravel()

    # Full KRR
    krr = RidgeMKL(lbd=lbd)
    krr.fit([K], y, holdout=te)
    yp_krr = krr.predict(te)

    # Calculate evar
    for m in models:
        for i, r in enumerate(rank_range):
            if m == "lars":
                yp = lars_kernel_predict(X=X[te], K=K_tr, act=act, Q=Q, R=R, beta=path[r]).ravel()
            elif m == "lars_ls":
                yp = Qt[:, :r].dot(Q[:, :r].T).dot(y[tr]).ravel()
            elif m == "lars_no":
                beta = np.linalg.lstsq(K_tr[:, act[:r]], y[tr])[0]
                yp = K_tr(X[te], X[tr])[:, act[:r]].dot(beta).ravel()
            elif m == "KRR":
                yp = yp_krr
            elif m == "icd":
                icd = RidgeLowRank(lbd=lbd, rank=r, method="icd")
                icd.fit([K_tr], y[tr])
                yp = icd.predict([X[te]])
            else:
                raise ValueError(m)
            evar[m][i] = (np.var(y[te]) - np.var(yp - y[te])) / np.var(y[te])

    i_cp = np.argmin(Cp)
    i_mse = np.argmin(mse)

    # Plots
    # plot_path(path)

    # Plot fit - training
    plt.figure()
    plt.plot(mu, "-", label="LARS")
    plt.plot(y[tr], ".", label="data")
    plt.title("%s (training)" % dataset)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "fit_training_%s.pdf" % dataset), bbox_inches="tight")
    plt.close()

    # Plot fit - test
    yp = lars_kernel_predict(X=X[te], K=K_tr, act=act, Q=Q, R=R, beta=path[i_cp])
    plt.figure()
    plt.plot(yp, "-", label="LARS")
    plt.plot(y[te], ".", label="data")
    plt.title("%s (test)" % dataset)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "fit_test_%s.pdf" % dataset), bbox_inches="tight")
    plt.close()

    # Risk estimation
    plt.figure()
    plt.plot(Cp)
    plt.plot(i_cp, Cp[i_cp], "v", color="green")
    plt.plot(i_mse, Cp[i_mse], "v", color="red")
    plt.xlabel("Model capacity $\\rightarrow$")
    plt.ylabel("Risk")
    plt.grid()
    plt.savefig(os.path.join(out_dir, "risk_%s.pdf" % dataset), bbox_inches="tight")
    plt.close()

    # Test MSE
    plt.figure()
    plt.plot(mse)
    plt.plot(i_cp, mse[i_cp], "v", color="green")
    plt.plot(i_mse, mse[i_mse], "v", color="red")
    plt.xlabel("Model capacity $\\rightarrow$")
    plt.ylabel("Test MSE")
    plt.grid()
    plt.savefig(os.path.join(out_dir, "mse_%s.pdf" % dataset), bbox_inches="tight")
    plt.close()

    # Test Expl. var
    plt.figure()
    for m in models:
        plt.plot(evar[m], label="%s (%.2f)" % (m, max(evar[m])), color=colors[m])
    plt.plot(i_cp, evar["lars"][i_cp], "v", color="green")
    plt.plot(i_mse, evar["lars"][i_mse], "v", color="red")
    plt.xlabel("Model capacity $\\rightarrow$")
    plt.ylabel("Test Expl. var")
    plt.ylim(0, 1.05)
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(out_dir, "evar_%s.pdf" % dataset), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for dset in KEEL_DATASETS:
        try:
            process(dset)
        except Exception as e:
            print("Exception with %s: %s" % (dset, e.message))

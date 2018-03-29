import numpy as np
import matplotlib.pyplot as plt
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.regression.ridge import RidgeMKL
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import ShuffleSplit


hlp = """
    Estimate variance of the unknown dataset. 
    Use L2-KRR with cross-validation to equalize the 
    residual variance on the training and the test set.
    
    Should be run with appropriately small data sets.
"""


def estimate_variance_cv(Ks, y, n_splits=10, n_lbd=13, lbd_min=-10, lbd_max=2):
    """ Estimate variance with cross-validation.
        Find the point where the training and test variance are most
        similar and return their mean. """

    S_tr = np.zeros((n_splits, n_lbd))
    S_te = np.zeros((n_splits, n_lbd))
    ss = ShuffleSplit(n_splits=n_splits)
    lbd_range = np.logspace(lbd_min, lbd_max, n_lbd)

    for si, (tr, te) in enumerate(ss.split(X=Ks[0].data, y=y)):
        for li, lbd in enumerate(lbd_range):
            model = RidgeMKL(method="l2krr", lbd=lbd)
            model.fit(Ks, y, holdout=te)
            yp_tr = model.predict(tr)
            yp_te = model.predict(te)

            S_tr[si, li] = np.var(y[tr].ravel() - yp_tr.ravel())
            S_te[si, li] = np.var(y[te].ravel() - yp_te.ravel())

    s1 = S_tr.mean(axis=0)
    s2 = S_te.mean(axis=0)

    i = np.argmin(np.absolute(s1 - s2))
    if i == 0 or i == len(s1) - 1:
        raise ValueError("Sigma estimate is outside the set lambda boundaires (i=%d)!" % i)

    est = np.mean([s1[i], s2[i]])
    return S_tr, S_te, est


def plot_variance_cv(S_tr, S_te, log=False):
    """ Plot relation between training and test estimates. """

    n = S_tr.shape[1]
    if log:
        f = "log10 "
        s1m = np.log10(S_tr).mean(axis=0)
        s2m = np.log10(S_te).mean(axis=0)
        s1e = np.log10(S_tr).std(axis=0)
        s2e = np.log10(S_te).std(axis=0)
    else:
        f = ""
        s1m = S_tr.mean(axis=0)
        s2m = S_te.mean(axis=0)
        s1e = S_tr.std(axis=0)
        s2e = S_te.std(axis=0)

    plt.figure()
    plt.errorbar(np.arange(n), s1m, yerr=s1e, label="Training")
    plt.errorbar(np.arange(n), s2m, yerr=s2e, label="Test")
    plt.xlabel("log10 $\\lambda$")
    plt.ylabel("%s$\\sigma_{est}$" % f)
    plt.legend()
    plt.grid()


def test_sim():
    """ Test on simulated data. """
    np.random.seed(42)
    N = 100
    noise = 0.01
    sigma_range = np.linspace(0.1, 1.0, 10) * N

    # Ground truth kernels
    X = np.linspace(-N, N, N).reshape((N, 1))
    Ks = [Kinterface(data=X,
                     kernel=exponential_kernel,
                     kernel_args={"sigma": sigma})
          for sigma in sigma_range]

    keff = len(Ks)/2
    f = mvn.rvs(mean=np.zeros((N,)), cov=Ks[keff][:, :])
    y = mvn.rvs(mean=f, cov=noise * np.eye(N))

    S_tr, S_te, s_est = estimate_variance_cv(Ks, y)
    s1 = S_tr.mean(axis=0)
    s2 = S_te.mean(axis=0)
    estimates = np.vstack((s1, s2)).mean(axis=0)
    assert s_est < estimates[0]
    assert s_est < estimates[1]

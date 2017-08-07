import os
import csv
import scipy.stats as st
import datetime

from scipy.linalg import sqrtm
from collections import Counter
from mklaren.kernel.string_kernel import *
from mklaren.kernel.string_util import *
from mklaren.kernel.kernel import kernel_sum
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from examples.string.string_lengthscales import generic_function_plot

# Experimental parameters
rank = 3
delta = 10
lbd = 0
L = 30
trueK = 4
max_K = 10
K_range = range(1, max_K+1)
normalize = False
n_tr = 500
n_te = 5000
N = n_tr + n_te
cv_iter = range(30)
ntarg = 1000

# Fixed output
# Create output directory
d = datetime.datetime.now()
dname = os.path.join("..", "output", "string_lengthscales_cv",
                     "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname): os.makedirs(dname)
rcnt = len(os.listdir(dname))
fname = os.path.join(dname, "results_%d.csv" % rcnt)
print("Writing to %s ..." % fname)

# Output
header = ["n", "L", "method", "rank", "iteration", "sp.corr", "sp.pval", "evar"]
fp = open(fname, "w", buffering=0)
writer = csv.DictWriter(fp, fieldnames=header)
writer.writeheader()


for cv in cv_iter:

    # Random subset of N sequences of length L
    X, _ = generate_data(N=N, L=L, p=0.0, motif="TGTG", mean=0, var=3)
    X = np.array(X)

    # Split into training in test set
    inxs = np.arange(N, dtype=int)
    np.random.shuffle(inxs)
    tr = inxs[:n_tr]
    te = inxs[n_tr:]
    X_tr = X[tr]
    X_te = X[te]

    # Generate a sparse signal based on 4-mer composion (maximum lengthscale)
    act = np.random.choice(tr, size=rank, replace=False)
    K_full = Kinterface(data=X, kernel=string_kernel, kernel_args={"mode": SPECTRUM, "K": trueK},
                   row_normalize=normalize)
    K_act = K_full[:, act]
    H = K_act.dot(sqrtm(np.linalg.inv(K_act[act])))
    w = st.multivariate_normal.rvs(mean=np.zeros((rank,)), cov=np.eye(rank))
    y = H.dot(w)
    y_tr = y[tr]
    y_te = y[te]

    # Proposal kernels
    args = [{"mode": SPECTRUM, "K": k} for k in K_range]
    Ksum = Kinterface(data=X_tr, kernel=kernel_sum,
                          row_normalize=normalize,
                          kernel_args={"kernels": [string_kernel] * len(args),
                                       "kernels_args": args})
    Ks = [Kinterface(data=X_tr, kernel=string_kernel,
                     kernel_args=a, row_normalize=normalize) for a in args]

    # Mklaren
    mklaren = Mklaren(rank=rank, delta=delta, lbd=lbd)
    mklaren.fit(Ks, y_tr)
    yp_mkl = mklaren.predict([X_te]*len(args)).ravel()
    mklaren_kernels = [(args[int(ky)]["K"], val) for ky, val in sorted(Counter(mklaren.G_mask).items())]
    for lg, num in sorted(mklaren_kernels, key=lambda t:t[1], reverse=True):
        print "K: %d (%d)" % (lg, num)

    # CSI
    csi = RidgeLowRank(rank=rank, method="csi",
                       method_init_args={"delta": delta}, lbd=lbd)
    csi.fit([Ksum], y_tr)
    yp_csi = csi.predict([X_te]).ravel()


    # Spearman correlation fo the fit
    sp_mkl = st.spearmanr(y_te, yp_mkl)
    sp_csi = st.spearmanr(y_te, yp_csi)
    print "\nMklaren fit: %.3f (%.5f)" % sp_mkl
    print "CSI fit: %.3f (%.5f)" % sp_csi

    # Explained variance of the model fit
    evar_mkl =  (np.var(y_te) - np.var(y_te - yp_mkl)) / np.var(y_te)
    evar_csi = (np.var(y_te) - np.var(y_te - yp_csi)) / np.var(y_te)

    rows = [{"n": N, "L": L, "method": "Mklaren", "rank": rank, "iteration": cv,
           "sp.corr": sp_mkl[0], "sp.pval": sp_mkl[1], "evar": evar_mkl},
           {"n": N, "L": L, "method": "CSI", "rank": rank, "iteration": cv,
           "sp.corr": sp_csi[0], "sp.pval": sp_csi[1], "evar": evar_csi}]
    writer.writerows(rows)


    # Plot a generic function plot
    fname = os.path.join(dname, "cv_K-%d_cv-%d.pdf" % (trueK, cv))
    generic_function_plot(f_out=fname, Ks=Ks, X=X_te,
                          models={"True": {"y": y_te, "color": "black", "fmt": "--", },
                                  "Mklaren": {"y": yp_mkl, "color": "green", "fmt": "-", },
                                  "CSI": {"y": yp_csi, "color": "red", "fmt": "-"}},
                          xlabel="K-mer length",
                          xnames=K_range,
                          truePar=K_range.index(trueK))
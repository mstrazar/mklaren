import os
import csv
import scipy.stats as st
import matplotlib.pyplot as plt
import datetime

from scipy.linalg import sqrtm
from collections import Counter
from mklaren.kernel.string_kernel import *
from mklaren.kernel.string_util import *
from mklaren.kernel.kernel import kernel_sum, kernel_to_distance
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank

# Experimental parameters
rank = 3
delta = 10
lbd = 0
L = 30
N = 50
trueK = 4
max_K = 10
K_range = range(1, max_K+1)
normalize = False
n_tr = 500
n_te = 5000
N = n_tr + n_te
cv_iter = range(10)

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
header = ["n", "L", "method", "rank", "iteration", "sp.corr", "sp.pval"]
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
    print "\nMklaren fit: %.3f (%.5f)"
    print "CSI fit: %.3f (%.5f)"

    rows = [{"n": N, "L": L, "method": "Mklaren", "rank": rank, "iteration": cv,
           "sp.corr": sp_mkl[0], "sp.pval": sp_mkl[1]},
           {"n": N, "L": L, "method": "CSI", "rank": rank, "iteration": cv,
           "sp.corr": sp_csi[0], "sp.pval": sp_csi[1]}]
    writer.writerows(rows)

    # Plot some sort of correlation between similarity for different K (inverse kernel matrix)
    # and the distance in the output prediction - limitation of the model
    # Caution, these kernel matrices are not independent
    corrs_mkl = []
    corrs_csi = []
    corrs_tru = []
    for ki in range(len(Ks)):
        Da = np.diag(Ks[ki].kernel(X[act], X[act], **Ks[ki].kernel_args)).reshape((rank, 1))
        Dt = np.array([Ks[ki].kernel(X[i], X[i], **Ks[ki].kernel_args) for i in te]).reshape((n_te, 1))
        Ki = Ks[ki].kernel(X_te, X[act], **Ks[ki].kernel_args)
        Di = -2 * Ki + Da.T + Dt

        Y = np.absolute(np.hstack([y_te.reshape(n_te, 1)]*rank) - y[act].reshape((1, rank)))
        Y_mkl = np.absolute(np.hstack([yp_mkl.reshape(n_te, 1)] * rank) - y[act].reshape((1, rank)))
        Y_csi = np.absolute(np.hstack([yp_csi.reshape(n_te, 1)] * rank) - y[act].reshape((1, rank)))

        sp_mkl = st.pearsonr(Di.ravel(), Y_mkl.ravel())
        sp_csi = st.pearsonr(Di.ravel(), Y_csi.ravel())
        sp_tru = st.pearsonr(Di.ravel(), Y.ravel())
        corrs_mkl.append(sp_mkl[0])
        corrs_csi.append(sp_csi[0])
        corrs_tru.append(sp_tru[0])

    # Plot a summary figure
    fname = os.path.join(dname, "cv_K-%d_cv-%d.pdf" % (trueK, cv))
    plt.figure()
    plt.title("Fitting various lengthscales with kernel sum")
    plt.plot(K_range, corrs_mkl, ".-", color="green", label="Mklaren", linewidth=2)
    plt.plot(K_range, corrs_csi, ".-", color="blue", label="CSI", linewidth=2)
    plt.plot(K_range, corrs_tru, "--", color="black", label="True f(x)", linewidth=2)
    plt.xlabel("K-mer length")
    plt.ylabel("Pearson correlation $K(i, j)$, $|y_i-y_j|$")
    plt.grid("on")
    plt.plot((trueK, trueK), plt.gca().get_ylim(), "-", color="black", label="True scale")
    plt.legend()
    plt.savefig()
    plt.close()
    print "Written %s" % fname


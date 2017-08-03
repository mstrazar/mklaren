import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from itertools import product, combinations

from collections import Counter
from mklaren.kernel.string_kernel import *
from mklaren.kernel.string_util import *
from mklaren.kernel.kernel import kernel_sum
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank

rank = 3
delta = 10
lbd = 0
max_K = 10
L = 30
N = 100
trueK = 4

# Random subset of N sequences of length L
X, _ = generate_data(N=N, L=L, p=0.0, motif="TGTG", mean=0, var=3)
X = np.array(X)

# Generate a sparse signal based on 4-mer composion (maximum lengthscale)
K = Kinterface(data=X, kernel=string_kernel, kernel_args={"mode": SPECTRUM, "K": trueK})
alpha = np.sort(np.random.rand(N, 1)**6, axis=0)
y = st.zscore(K[:, :].dot(alpha))
yr = y.ravel()

# Proposal kernels
args = [{"mode": SPECTRUM, "K": k} for k in range(2, max_K)]
Ksum = Kinterface(data=X, kernel=kernel_sum,
                      kernel_args={"kernels": [string_kernel] * len(args),
                                   "kernels_args": args})
Ks = [Kinterface(data=X, kernel=string_kernel, kernel_args=a) for a in args]

# Mklaren
mklaren = Mklaren(rank=rank, delta=delta, lbd=lbd)
mklaren.fit(Ks, y)
yp_mkl = mklaren.predict([X]*len(args)).ravel()
mklaren_kernels = [(args[int(ky)]["K"], val) for ky, val in sorted(Counter(mklaren.G_mask).items())]
for lg, num in sorted(mklaren_kernels, key=lambda t:t[1], reverse=True):
    print "K: %d (%d)" % (lg, num)


# CSI
csi = RidgeLowRank(rank=rank, method="csi",
                   method_init_args={"delta": delta}, lbd=lbd)
csi.fit([Ksum], y)
yp_csi = csi.predict([X]).ravel()


# Prind data along with predictions
for xi, yi, ym, yc in sorted(zip(X, y, yp_mkl, yp_csi), key=lambda t: t[1]):
    print "%s\t%.3f\t%.3f\t%.3f" % (xi, yi, ym, yc)

# Spearman correlation fo the fit
print "\nMklaren fit: %.3f (%.5f)" % st.spearmanr(y, yp_mkl)
print "CSI fit: %.3f (%.5f)" % st.spearmanr(y, yp_csi)

# Represent fit on figure
plt.figure()
plt.plot(yr, yp_mkl, ".", color="green", label="Mklaren")
plt.plot(yr, yp_csi, ".", color="blue", label="CSI")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.legend()


# Spearman correlation fo the fit
print "\nMklaren residual corr.: %.3f (%.5f)" % st.spearmanr(yr, yr-yp_mkl,)
print "CSI residual corr.: %.3f (%.5f)" % st.spearmanr(yr, yr-yp_csi)
print

# Residual graph
plt.figure()
plt.plot(yr, yr-yp_mkl, ".", color="green", label="Mklaren")
plt.plot(yr, yr-yp_csi, ".", color="blue", label="CSI")
plt.xlabel("True")
plt.ylabel("Residual")
plt.legend()

# Plot some sort of correlation between similarity for different K (inverse kernel matrix)
# and the distance in the output prediction - limitation of the model
# Caution, these kernel matrices are not independent
corrs_mkl = []
corrs_csi = []
for ki in range(len(Ks)):
    Ki = Ks[ki]
    ks = np.array([-Ki[i, j] for i, j in combinations(range(N), 2)]).ravel()
    ys_mkl = np.array([abs(yp_mkl[i] - yp_mkl[j]) for i, j in combinations(range(N), 2)])
    ys_csi = np.array([abs(yp_csi[i] - yp_csi[j]) for i, j in combinations(range(N), 2)])
    sp_mkl = st.spearmanr(ks, ys_mkl)
    sp_csi = st.spearmanr(ks, ys_csi)
    corrs_mkl.append(sp_mkl[0])
    corrs_csi.append(sp_csi[0])
    print "\nK=%d" % Ki.kernel_args["K"]
    print "\tMklaren kernel corr.: %.3f (%.5f)" % sp_mkl
    print "\tCSI kernel corr.: %.3f (%.5f)" % sp_csi

# Plot a summary figure
plt.figure()
plt.title("Fitting various lengthscales with kernel sum")
plt.plot(range(2, max_K), corrs_mkl, ".-", color="green", label="Mklaren", linewidth=2)
plt.plot(range(2, max_K), corrs_csi, ".-", color="blue", label="CSI", linewidth=2)
plt.xlabel("K-mer length")
plt.ylabel("Sp. correlation $y_i$, $y_j$")
plt.grid("on")
plt.plot((trueK, trueK), plt.gca().get_ylim(), "-", color="black", label="True scale")
plt.legend()
plt.savefig("/Users/martin/Dev/mklaren/examples/output/string/lengthscales_%d.pdf" % trueK)
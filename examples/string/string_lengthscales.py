import scipy.stats as st
import matplotlib.pyplot as plt

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

# Random subset of N sequences of length L
X, _ = generate_data(N=N, L=L, p=0.0, motif="TGTG", mean=0, var=3)
X = np.array(X)

# Generate a sparse signal based on 4-mer composion (maximum lengthscale)
K = Kinterface(data=X, kernel=string_kernel, kernel_args={"mode": SPECTRUM, "K": trueK},
               row_normalize=normalize)
y = st.multivariate_normal.rvs(mean=np.zeros((N,)), cov=K[:, :]).reshape((N, 1))
yr = y.ravel()

# Proposal kernels
args = [{"mode": SPECTRUM, "K": k} for k in K_range]
Ksum = Kinterface(data=X, kernel=kernel_sum,
                      row_normalize=normalize,
                      kernel_args={"kernels": [string_kernel] * len(args),
                                   "kernels_args": args})
Ks = [Kinterface(data=X, kernel=string_kernel, kernel_args=a, row_normalize=normalize) for a in args]

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


# Print data along with predictions
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
corrs_tru = []
for ki in range(len(Ks)):
    Ki = Ks[ki][:, :]
    ks = np.array([-Ki[i, j] for i, j in combinations(range(N), 2)]).ravel()
    Di = kernel_to_distance(Ks[ki])
    ks = np.array([Di[i, j] for i, j in combinations(range(N), 2)]).ravel()
    ys_mkl = np.array([abs(yp_mkl[i] - yp_mkl[j]) for i, j in combinations(range(N), 2)]).ravel()
    ys_csi = np.array([abs(yp_csi[i] - yp_csi[j]) for i, j in combinations(range(N), 2)]).ravel()
    ys_tru = np.array([abs(y[i] - y[j]) for i, j in combinations(range(N), 2)]).ravel()
    sp_mkl = st.pearsonr(ks, ys_mkl)
    sp_csi = st.pearsonr(ks, ys_csi)
    sp_tru = st.pearsonr(ks, ys_tru)
    corrs_mkl.append(sp_mkl[0])
    corrs_csi.append(sp_csi[0])
    corrs_tru.append(sp_tru[0])


# Plot a summary figure
plt.figure()
plt.title("Fitting various lengthscales with kernel sum")
plt.plot(K_range, corrs_mkl, ".-", color="green", label="Mklaren", linewidth=2)
plt.plot(K_range, corrs_csi, ".-", color="blue", label="CSI", linewidth=2)
plt.plot(K_range, corrs_tru, "--", color="black", label="True f(x)", linewidth=2)
plt.xlabel("K-mer length")
plt.ylabel("Pearson correlation $K(i, j)$, $|y_i-y_j|$")
plt.grid("on")
plt.plot((trueK, trueK), plt.gca().get_ylim(), "-", color="black", label="True scale")
plt.ylim(0, max(max(corrs_mkl), max(corrs_csi)))
plt.legend()
plt.savefig("/Users/martin/Dev/mklaren/examples/output/string/lengthscales_%d_1.pdf" % trueK)
from mklaren.projection.icd import ICD
from mklaren.util.la import fro_prod
from mklaren.kernel.kernel import center_kernel
from scipy.stats import pearsonr, spearmanr
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pyplot as plt
import itertools as it


# Do pivots tend to represent some portions of space better than others, or is the
# relation purely random?

# Does this transfer to the PSD cone of kernels?

n = 1000
rank = 100

# Basically sum of rank one kernels, tu can be arbitrarily big.
X = np.random.rand(n, rank)
K = X.dot(X.T)

# Fit an ICD model
icd = ICD(rank=rank)
icd.fit(K)
G = icd.G

# Correlation matrix
C = np.zeros((rank, rank))
for i, j in it.product(range(rank), range(rank)):
    C[i, j] = pearsonr(X[:, i], G[:, j])[0]

# Cluster correlation matrix
L1 = sch.linkage(C)
inxs1 = sch.dendrogram(L1, no_plot=True)["leaves"]
L2 = sch.linkage(C.T)
inxs2 = sch.dendrogram(L2, no_plot=True)["leaves"]

C = C[inxs1, :]
C = C[: ,inxs2]

plt.figure()
plt.pcolor(C)
plt.xlabel("Pivot")
plt.ylabel("Kernel")
plt.show()



# Find best pivot for each kernel - SHIFTED
counter = np.zeros((rank,))
for i in range(rank):
    cors = [pearsonr(X[:, i], G[:, j])[0] for j in range(rank)]
    jm = np.argmax(cors)
    ba = np.max(cors[0:jm] + cors[jm+1:])
    counter[jm] += 1
    print("Kernel %d, best pivot match: %d, cor: %f, best other: %f" %(i, jm, cors[jm], ba))

# This distribution is heavily shifted to the left
plt.figure()
plt.bar(range(rank), counter)
plt.xlabel("Pivot")
plt.ylabel("Number of kernel matches")


# Find best kernel for each pivot - UNIFORM
counter = np.zeros((rank,))
for i in range(rank):
    cors = [pearsonr(G[:, i], X[:, j])[0] for j in range(rank)]
    jm = np.argmax(cors)
    ba = np.max(cors[0:jm] + cors[jm+1:])
    counter[jm] += 1
    print("Pivot %d, best kernel match: %d, cor: %f, best other: %f" %(i, jm, cors[jm], ba))

# This distribution is heavily shifted to the left
plt.figure()
plt.bar(range(rank), sorted(counter, reverse=True))
plt.xlabel("Kernel")
plt.ylabel("Number of pivot matches")
plt.show()




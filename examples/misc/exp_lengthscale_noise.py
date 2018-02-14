hlp = """
    Investigate the interplay between noise and lengthscale.
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from mklaren.kernel.kernel import exponential_kernel

n = 101
X = np.linspace(-10, 10, n).reshape(n, 1)
K = exponential_kernel(X, X, gamma=1)

# Generate a random smooth function
w = np.random.randn(n)
f = K.dot(w)
f = f - f.mean()
eps = np.random.randn(n)

t = 50
N = 30
R = np.zeros((N, N))
gamma_range = np.linspace(0.1, 5.0, N)
noise_range = np.linspace(0.1, 10.0, N)

for i, j in it.product(range(N), range(N)):
    y = f + noise_range[i] * eps
    L = exponential_kernel(X, X, gamma=gamma_range[j])
    R[i, j] = np.absolute(L[:, i].dot(y.ravel()) / np.linalg.norm(L[:, i]))

# Heatmap
plt.figure()
plt.pcolor(R)
plt.xlabel("Frequency")
plt.ylabel("Noise")
plt.colorbar()
plt.show()

# Draw frequency at noise point
p = 0
r0 = R[p, :]
r1 = R[N-1, :]
plt.figure()
plt.plot([1, 1], [min(r0), max(r0)], "k-")
plt.plot(gamma_range, r0, label="noise = 0")
plt.plot(gamma_range, r1, label="noise > 0")
plt.xlabel("Frequency")
plt.ylabel("Correlation")
plt.legend()
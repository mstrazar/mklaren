hlp = """
    Investigate the shapes of bisectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from mklaren.kernel.kernel import exponential_kernel
from examples.lars_vs_greedy.mklaren2 import find_bisector

n = 101
X = np.linspace(-10, 10, n).reshape(n, 1)
K = exponential_kernel(X, X, gamma=0.3)

i = 20
j = n - i - 1
k = 51

u, A = find_bisector(K[:, [i, j, k]])

plt.figure()
plt.plot(X.ravel(), K[:, i], "--", color="blue", linewidth=0.5)
plt.plot(X.ravel(), K[:, j], "--", color="blue", linewidth=0.5)
plt.plot(X.ravel(), K[:, k], "--", color="blue", linewidth=0.5)
plt.plot(X.ravel(), u.ravel(), color="green")
plt.show()

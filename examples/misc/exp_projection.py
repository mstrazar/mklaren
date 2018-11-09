hlp = """
    Investigate the projection size.
"""
import numpy as np
import matplotlib.pyplot as plt
from mklaren.kernel.kernel import exponential_kernel

n = 101
X = np.linspace(-10, 10, n).reshape(n, 1)
K = exponential_kernel(X, X, gamma=1)

# Generate a random smooth function
w = np.random.randn(n)
f = K.dot(w)
f = f - f.mean()

# Project on one basis function
i = 80
p = K[:, i] * f
pnz = np.where(np.absolute(p) > 1e-2)[0]

# Plot
plt.figure()
plt.plot(X.ravel(), K[:, i], "--", color="blue", linewidth=0.5)
plt.plot(X.ravel(), f.ravel(), "--", color="black", linewidth=0.5)
plt.plot(X.ravel(), p.ravel(), color="red", linewidth=0.2)
plt.plot(X.ravel()[pnz], p.ravel()[pnz], color="red", linewidth=1.5)
plt.show()

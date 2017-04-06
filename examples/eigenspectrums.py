"""
Comparison of kernel eigenspectrums in different situations.
Investigate various situations that can give rise to overfitting.

"""
import numpy as np
import matplotlib.pyplot as plt

from mklaren.kernel.kernel import exponential_kernel, linear_kernel, poly_kernel
from mklaren.kernel.kinterface import Kinterface


# Complete data range
p = 10
n = 5
Xt = np.linspace(-2, 2, 2 * n).reshape((2 * n, 1))


# Eigenvalue of the exponential kernels
# Shorter length-scales (sigmas) converge to identity
plt.figure()
for gi, g in enumerate(np.logspace(-1, 3, 4)):
    K = Kinterface(data=Xt, kernel=exponential_kernel, kernel_args={"gamma": g})
    vals, vecs = np.linalg.eig(K[:, :])
    plt.plot(vals, label="$\gamma=%0.2f,\  \sigma^2=%0.4f$" % (g, 1.0/g), linewidth=2)
plt.legend()
plt.xlabel("Eigenvalue index")
plt.ylabel("Magnitutde")
plt.title("Exponentiated quadratic")
plt.savefig("examples/output/eigenspectrums/exponential.pdf")
plt.close()

# Random kernels converge to identity (pure-noise) covariance
# Important: data must be sampled from normal rather than normal distribution
n = 5
plt.figure()
for pi, p in enumerate(np.logspace(1, 6, 6)):
    Y = np.random.randn(n, p)
    K = Kinterface(data=Y, kernel=linear_kernel, row_normalize=True)
    vals, vecs = np.linalg.eig(K[:, :])
    vals = sorted(vals, reverse=True)
    plt.plot(vals, label="p=%d" % p, linewidth=2)

    print("p=%d"% p)
    print(K[:, :])
    print()
plt.legend()
plt.xlabel("Eigenvalue index")
plt.ylabel("Magnitutde")
plt.title("Linear kernel (p: dimensionality of space)")
plt.savefig("examples/output/eigenspectrums/linear.pdf")
plt.show()
"""
Comparison of kernel eigenspectrums in different situations.
Investigate various situations that can give rise to overfitting.

"""
import numpy as np
import matplotlib.pyplot as plt

from mklaren.kernel.kernel import exponential_kernel, linear_kernel, poly_kernel, periodic_kernel
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


# Image of the periodic kernel.
X = np.linspace(-5, 5, 100).reshape((100, 1))
plt.figure()
for l in [1, 3, 10]:
    k = periodic_kernel(X, np.array([[0]]), l=l, )
    plt.plot(X.ravel(), k.ravel(), label=str(l), linewidth=np.log10(l)+1)
plt.legend(title="lengthscale")
plt.xlabel("x")
plt.ylabel("k(x, 0)")
plt.savefig("/Users/martin/Dev/mklaren/examples/output/eigenspectrums/periodic_lengthscale.pdf")
plt.close()


# Image of the periodic kernel.
X = np.linspace(-5, 5, 100).reshape((100, 1))
plt.figure()
for l in [1, 3, 10]:
    k = periodic_kernel(X, np.array([[0]]), sigma=l, )
    plt.plot(X.ravel(), k.ravel(), label=str(l), linewidth=np.log10(l)+1)
plt.legend(title="sigma")
plt.xlabel("x")
plt.ylabel("k(x, 0)")
plt.savefig("/Users/martin/Dev/mklaren/examples/output/eigenspectrums/periodic_sigma.pdf")
plt.close()

# Covariance structure of periodic kernel
eigs = []
X = np.linspace(-5, 5, 100).reshape((100, 1))
for per in [1, 3, 10]:
    K = Kinterface(data=X, kernel=periodic_kernel, row_normalize=True, kernel_args={"per": per})
    eig, _ = np.linalg.eig(K[:, :])
    eigs.append((per, eig))

# Plot eigenspectrums
plt.figure()
for p, eig in eigs:
    plt.plot(sorted(eig, reverse=True), label=str(p), linewidth=np.log10(p)+1,  alpha=0.5)
plt.legend(loc=1, title="period")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.title("Periodic kernel")
plt.grid("on")
plt.savefig("/Users/martin/Dev/mklaren/examples/output/eigenspectrums/periodic.pdf")
plt.close()
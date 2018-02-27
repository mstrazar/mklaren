import numpy as np
import matplotlib.pyplot as plt
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.projection.icd import ICD

hlp = """
    Experiments with lower bound on which inducing point to sample next.
    This is similar than approximate leverage scores for an unsupervised problems of approximating the kernel.

    Is it possible to instead work with the orthogonal subspace with respect to some y?  
        Yes, but only for look-ahead type approximation.
        
    Is it possible to compute risk given that we actually have an approximation?
        Perhaps compare with the values that you already have checked.
    
    Define the approximate matrix.    
        L = GG'.
    Approximation is exact for L[A, A] == K[A, A], where A is the active set.
    However, it is not exact for L[A, Aj] != K[A, Aj] where A != Aj.
    Adding a point, Aj = (A union j), makes L[Aj, Aj] == K[Aj, Aj]
    
    All values K[:, i] have to be computed at iteration i.
    The values K[:, i] - GG'[:, :i] are not used for anything and represent "how much we already approximated". 
        Could they be used to complement the lower bound.
        
"""

# Generate data
n = 100
X = np.linspace(-10, 10, n).reshape((n, 1))
w = np.random.randn(n)
K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.3})
y = K[:, :].dot(w)
y = y - y.mean()

# Fit model
rank = 30
model = ICD(rank=rank, mode=ICD.MODE_RANDOM)
model.fit(K)
model_norm = ICD(rank=rank, mode=ICD.MODE_NORM)
model_norm.fit(K)

# Plot the fit
# plt.figure()
# plt.plot(X.ravel(), y.ravel(), "-")
# for k in range(model.rank):
#     ki = model.active_set_[k]
#     plt.plot(X.ravel(), model.D[:, k], "c-")
#     plt.plot(X.ravel()[ki], model.D[ki, k], "r.")
#     plt.text(X.ravel()[ki], model.D[ki, k]+1, str(k))
# plt.xlabel("x")
# plt.ylabel("f(x)")

# Compare true gain and approximate gain
G = model.G
lb = [model.D[model.active_set_[k], k] for k in range(model.rank)]
gain = [np.linalg.norm(G[:, :k+1].dot(G[:, :k+1].T) - G[:, :k].dot(G[:, :k].T), ord=1) for k in range(model.rank)]
error = [np.linalg.norm(K[:, :] - G[:, :k].dot(G[:, :k].T), ord=1) for k in range(model.rank)]

# Compare true gain and approximate gain
G = model_norm.G
lb = [model_norm.D[model_norm.active_set_[k], k] for k in range(model_norm.rank)]
gain_norm = [np.linalg.norm(G[:, :k+1].dot(G[:, :k+1].T) - G[:, :k].dot(G[:, :k].T), ord=1) for k in range(model_norm.rank)]
error_norm = [np.linalg.norm(K[:, :] - G[:, :k].dot(G[:, :k].T), ord=1) for k in range(model_norm.rank)]
#
# plt.figure()
# plt.plot(lb, label="Lower bound")
# plt.plot(gain, "--", label="Gain (random)")
# plt.plot(gain_norm, label="Gain (norm)")
# plt.xlabel("Iteration")
# plt.ylabel("Gain")
# plt.grid()
# plt.legend()

plt.figure()
plt.plot(error, "--", label="Error (random)")
plt.plot(error_norm, label="Error (norm)")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.grid()
plt.legend()

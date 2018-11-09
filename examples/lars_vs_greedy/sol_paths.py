hlp = """ Reduction in the error as the number of iterations increases. """

import numpy as np
import matplotlib.pyplot as plt
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from examples.lars_vs_greedy.mklaren2 import MklarenNyst


# Generate data
n = 100
gamma = 0.3
noise = 10
X = np.linspace(-10, 10, n).reshape((n, 1))
K = exponential_kernel(X, X, gamma=gamma)
w = np.random.randn(n, 1)
f = K.dot(w) / K.dot(w).mean()
noise_vec = np.random.randn(n, 1)  # Crucial: noise is normally distributed
y = f + noise * noise_vec

# Model
gamma_range = [0.01, 0.03, 0.1, 0.3, 1.0]
Ks = [Kinterface(data=X,
                 kernel=exponential_kernel,
                 kernel_args={"gamma": gam}) for gam in gamma_range]
rank = 10
model_greedy = MklarenNyst(rank=rank)
model_greedy.fit_greedy(Ks, y)
model_lars = MklarenNyst(rank=rank)
model_lars.fit(Ks, y)


# Plot
plt.close("all")
for name, model in zip(("LARS", "Greedy"), (model_lars, model_greedy)):
    plt.figure()
    plt.title(name)
    plt.plot(X.ravel(), y.ravel(), ".")
    plt.plot(X.ravel(), f.ravel(), "-")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    for si, sp in enumerate(model.sol_path):
        plt.plot(X.ravel(), sp.ravel(), "r-", linewidth=(1 + si), alpha=0.2)
    for pi, (q, i) in enumerate(model.active):
        plt.text(X[i], 0, "%d" % pi)


# Error vs. rank
errors = {
    "LARS": np.zeros((rank,)),
    "Greedy": np.zeros((rank,))
}
for name, model in zip(("LARS", "Greedy"), (model_lars, model_greedy)):
    for si, sp in enumerate(model.sol_path):
        errors[name][si] = np.linalg.norm(sp-f)

# Plot
plt.figure()
for name, model in zip(("LARS", "Greedy"), (model_lars, model_greedy)):
    plt.plot(errors[name], label=name)
plt.grid()
plt.legend()
plt.xlabel("Rank")
plt.ylabel("$\|\mu - f\|$")

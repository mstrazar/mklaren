import numpy as np
import matplotlib.pyplot as plt
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface
from examples.lars_vs_greedy.mklaren2 import MklarenNyst
from sklearn.linear_model.ridge import Ridge


hlp = """
    Compare regularization paths for Mklaren and L2 Ridge regression.
"""

# Generate data
n = 100
gamma = 0.3
noise = 3.0
X = np.linspace(-30, 30, n).reshape((n, 1))
K = exponential_kernel(X, X, gamma=gamma)
w = np.random.randn(n, 1)
f = K.dot(w) / K.dot(w).mean()
noise_vec = np.random.randn(n, 1)  # Crucial: noise is normally distributed
y = f + noise * noise_vec


# Model
gamma_range = [gamma]
Ks = [Kinterface(data=X,
                 kernel=exponential_kernel,
                 kernel_args={"gamma": gam}) for gam in gamma_range]
# LARS
rank = 20
model_lars = MklarenNyst(rank=rank)
model_lars.fit(Ks, y)
path_lars = [np.var(y - mu) for mu in model_lars.sol_path ]
path_lars_f = [np.var(f - mu) for mu in model_lars.sol_path ]

# 'Continuous' regularization path
sol_path = [np.zeros((n, 1))] + model_lars.sol_path
soft = np.linspace(0, 1, 50)
P = np.zeros((len(soft) * (rank - 1), n))
for i, (sp, bsk, g) in enumerate(zip(sol_path, model_lars.bisector_path, model_lars.grad_path)):
    a = i * len(soft)
    b = (i+1) * len(soft)
    P[a:b, :] = (sp + (soft * g) * bsk).T

path_cont = np.array([np.var(P[i, :].ravel() - y.ravel()) for i in range(len(P))])
path_cont_f = np.array([np.var(P[i, :].ravel() - f.ravel()) for i in range(len(P))])
inxs = range(0, len(path_cont), len(soft))

# Plot path and sample points
plt.figure()
plt.plot(range(len(path_cont)), path_cont, "-", color="blue", label="$\|y - h(x)\|$")
plt.plot(inxs, path_cont[inxs], ".", color="blue")
plt.plot(range(len(path_cont_f)), path_cont_f, "-", color="orange", label="$\|f(x) - h(x)\|$")
plt.plot(inxs, path_cont_f[inxs], ".", color="orange")
plt.xlabel("Model capacity $\\rightarrow$")
plt.ylabel("MSE")
plt.legend()


# Ridge regression
lbd_range = np.linspace(0.01, 30, 20)[::-1]
path_ridge = [np.var(y.ravel() - Ridge(alpha=lbd).fit(Ks[0][:, :], y).predict(Ks[0][:, :]).ravel())
              for lbd in lbd_range]
path_ridge_f = [np.var(f.ravel() - Ridge(alpha=lbd).fit(Ks[0][:, :], y).predict(Ks[0][:, :]).ravel())
                for lbd in lbd_range]


if False:
    # Compare with ridge - fit MSE
    plt.figure()
    plt.plot(path_ridge, label="Ridge")
    plt.plot(path_lars, label="LARS")
    plt.legend()
    plt.ylabel("$\|y-h(x)\|$")
    plt.xlabel("$\leftarrow$ $\lambda$")

    # Compare with ridge - true MSE
    plt.figure()
    plt.plot(path_ridge_f, label="Ridge")
    plt.plot(path_lars_f, label="LARS")
    plt.legend()
    plt.ylabel("$\|f(x)-h(x)\|$")
    plt.xlabel("$\leftarrow$ $\lambda$")

# Wait
plt.show()

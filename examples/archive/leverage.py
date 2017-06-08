# Unit test with example in Alaoui and Mahoney, 2009.
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import exponential_kernel
from mklaren.projection.nystrom import Nystrom

N = 100
gamma_range = np.logspace(-3, 1, 5)

xpd = st.expon()
x = xpd.rvs(N)
xt = np.hstack([x, x.max() - x])
xt = np.array(sorted(xt))
xt = xt.reshape((2 * N, 1))


# Exact and approximate leverage scores.
L_app = np.zeros((len(gamma_range), 2 * N))
L_exact = np.zeros((len(gamma_range), 2 * N))
for hi, g in enumerate(gamma_range):
    K = Kinterface(kernel=exponential_kernel, data=xt, kernel_args={"gamma": g})
    model = Nystrom(lbd=1, rank=10)
    lev = model.leverage_scores(K)
    L_app[hi, :] = lev

    model = Nystrom(lbd=1, rank=N)
    lev = model.leverage_scores(K)
    L_exact[hi, :] = lev

h = len(gamma_range)
fig, ax = plt.subplots(h, 2, figsize=(8, 2.2 * h))
for hi, g in enumerate(gamma_range):
    ax[hi, 0].hist(xt)
    ax[hi, 0].set_xlabel("Value")
    ax[hi, 0].set_ylabel("Density")
    ax[hi, 1].plot(xt, L_app[hi, :], color="blue")
    ax[hi, 1].plot(xt, L_exact[hi, :], color="red")
    ax[hi, 1].set_xlabel("Value")
    ax[hi, 1].set_ylabel("Leverage")
    ax[hi, 1].set_title("$\gamma=%.2f$" % g)
fig.tight_layout()
plt.show()
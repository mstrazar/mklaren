hlp = """
    Compare the eigenspectrum as adding more and more kernels.
    Implicit features are very correlated, leading to a fastly dropping eigenspectrum,
    even with adding more and more kernels.
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mklaren.kernel.string_kernel import *
from mklaren.kernel.string_util import *
from mklaren.kernel.kernel import kernel_sum
from mklaren.kernel.kinterface import Kinterface


# Generate data
X, y = generate_data(N=100, L=100, p=0.5, motif="TGTG", mean=0, var=3, seed=9)
Xa = np.array(X)
X_tr = Xa[:50]
X_te = Xa[50:]
y_tr = y[:50]
y_te = y[50:]

args = [
    {"mode": SPECTRUM, "K": 2},
    {"mode": SPECTRUM, "K": 3},
    {"mode": SPECTRUM, "K": 4},
    {"mode": SPECTRUM, "K": 5},
    {"mode": WD, "K": 2},
    {"mode": WD, "K": 4},
    {"mode": WD, "K": 5},
]

# Compare eigenspectrums depending on the number of included kernels
eigs = []
for p in range(1, len(args)+1):
    Ksum = Kinterface(data=X_tr, kernel=kernel_sum,
                      row_normalize=True,
                      kernel_args={"kernels": [string_kernel] * len(args[:p]),
                                       "kernels_args": args[:p]})

    eig, _ = np.linalg.eig(Ksum[:, :])
    eigs.append((p, eig))

# Plot eigenspectrums
plt.figure()
for p, eig in eigs:
    plt.plot(sorted(np.log(eig), reverse=True), label=str(p), linewidth=p,  alpha=0.5)
plt.legend(loc=3, title="Num. kernels")
plt.xlabel("Component")
plt.ylabel("Log eigenvalue")
plt.savefig("/Users/martin/Dev/mklaren/examples/output/string/eigenspectrum.pdf")
plt.close()



# Eigenspectrum depending on sequence length
l_range = (10, 100, 1000)
eigs = []
for L in l_range:
    X, y = generate_data(N=100, L=L, p=0.5, motif="TGTG", mean=0, var=3, seed=9)
    Ksum = Kinterface(data=np.array(X), kernel=kernel_sum,
                      row_normalize=True,
                      kernel_args={"kernels": [string_kernel] ,
                                   "kernels_args": [{"mode": WD, "K": 2}]})
    eig, _ = np.linalg.eig(Ksum[:, :])
    eigs.append((L, eig))


# Plot eigenspectrums
plt.figure()
for p, eig in eigs:
    plt.plot(sorted(eig, reverse=True), label=str(p), linewidth=np.log10(p),  alpha=0.5)
plt.legend(loc=1, title="Sequence length")
plt.xlabel("Component")
plt.ylabel("Eigenvalue")
plt.title("Weighted degree kernel")
plt.grid("on")
plt.savefig("/Users/martin/Dev/mklaren/examples/output/string/eigenspectrum_length.pdf")
plt.close()
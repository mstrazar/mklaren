# Kernels
import numpy as np
import matplotlib.pyplot as plt
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface


p_range = (3, 10, 30)
eigs = []

for p in p_range:
    X = np.random.rand(100, 3)
    gam_range = np.logspace(-5, 5, p, base=2)  # RBF kernel parameter
    Ksum = Kinterface(data=X,
                      kernel=kernel_sum,
                      kernel_args={"kernels": [exponential_kernel] * len(gam_range),
                                   "kernels_args": [{"gamma": gam} for gam in gam_range]})
    eig, _ = np.linalg.eig(Ksum[:, :])
    eigs.append(eig)


plt.figure()
for ei, e in enumerate(eigs):
    plt.plot(np.log(sorted(e, reverse=True)), label="p="+str(p_range[ei]), linewidth=2*(ei+1))
plt.legend()
plt.xlabel("Eigenvalue order")
plt.ylabel("Log magnitude")
plt.savefig("../output/delve_num_kernels/eigenspectrum.pdf")
plt.close()
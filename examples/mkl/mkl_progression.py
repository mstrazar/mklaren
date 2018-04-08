import matplotlib.pyplot as plt
import numpy as np
import os

# Kernels
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface

# New methods
from examples.lars.lars_mkl import LarsMKL
from examples.lars.lars_group import p_ri, p_const, p_sc
from examples.mkl.mkl_est_var import estimate_sigma_dist
from scipy.stats import multivariate_normal as mvn

hlp = """
    Compare the progression of kernel selection for multiple trials 
    depending on the cost function. We know that different cost functions
    persist on different kernels. Does this reflect on lengthscales.
"""

# Parameters
out_dir = "/Users/martins/Dev/mklaren/examples/mkl/output/mkl_progression"
N = 200
delta = 5
rank = 10
nk = 10
replicates = 100

# Variables
noise_range = np.logspace(-4, 1, 6)
noise_marks = [chr(97 + i) for i in range(len(noise_range))]
functions = [p_ri, p_sc, p_const]

# Auxillary
formats = {"lars-ri": "gv-",
           "lars-sc": "bv-",
           "lars-co": "cv-"}


def plot_progressions(Ps, nk=None, title=""):
    """ Plot progression as summary of multiple runs"""
    nk = Ps.max() + 1 if nk is None else nk
    C = np.zeros((nk, Ps.shape[1]))
    for P in Ps:
        C[P.astype(int), np.arange(len(P))] += 1
    D = C / C.sum(axis=0)
    plt.figure()
    plt.imshow(D)
    plt.xlabel("Step")
    plt.ylabel("Kernel")
    plt.title(title)


def process():

    # Generate input and kernel space
    X = np.linspace(-N, N, N).reshape((N, 1))
    sigma_range = estimate_sigma_dist(X, q=nk)
    Ks = [Kinterface(data=X,
                     kernel=exponential_kernel,
                     kernel_args={"sigma": sig}) for sig in sigma_range]

    # Effective kernel is a medium length scale
    Keff = Ks[len(sigma_range)/2]

    # Iterate
    for noise, nmark in zip(noise_range, noise_marks):
        for func in functions:
            # Replicates given a cost function
            Ps = np.zeros((replicates, rank))
            for repl in range(replicates):
                f = mvn.rvs(mean=np.zeros((N,)), cov=Keff[:, :])
                y = mvn.rvs(mean=f, cov=noise * np.eye(N))
                model = LarsMKL(rank=rank, delta=delta, f=func)
                model.fit(Ks, y)
                Ps[repl] = model.P

            # Plot
            fname = os.path.join(out_dir, "prog_%s_noise_%s.pdf" % (func.__name__, nmark))
            plot_progressions(Ps, nk, title="%s | log(noise) = %d" % (func.__name__, np.log10(noise)))
            plt.savefig(fname)
            plt.close()
            print("Written %s" % fname)


if __name__ == "__main__":
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Makedir %s" % out_dir)
    process()

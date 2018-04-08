import matplotlib.pyplot as plt
import itertools as it
import numpy as np
import os

# Kernels
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface

# New methods
from examples.lars.lars_mkl import LarsMKL
from examples.lars.lars_group import p_ri, p_const, p_sc
from examples.mkl.mkl_est_var import estimate_sigma_dist
from examples.mkl.mkl_progression import plot_progressions

# Datasets
from datasets.keel import load_keel, KEEL_DATASETS

hlp = """
    Compare the progression of kernel selection for multiple trials 
    depending on the cost function for real-world data. Does this reflect on lengthscales?.
"""

# Parameters
out_dir = "/Users/martins/Dev/mklaren/examples/mkl/output/mkl_keel_prog"
N = 300
delta = 5
rank = 20
nk = 10
replicates = 30

# Variables
functions = [p_ri, p_sc, p_const]

# Auxillary
formats = {"lars-ri": "gv-",
           "lars-sc": "bv-",
           "lars-co": "cv-"}


def load_data_sample(dataset, Nmax, nk=10, p_tr=0.6):
    """ Load a random sample of rows from dataset. """
    data = load_keel(dataset)
    Nall = len(data["data"])
    size = min(int(p_tr * Nall), Nmax)
    inxs = np.random.choice(np.arange(len(data["data"])), size, replace=False)
    X, y = data["data"][inxs], data["target"][inxs]
    sigma_range = estimate_sigma_dist(X, q=nk)
    Ks = [Kinterface(data=X,
                     kernel=exponential_kernel,
                     kernel_args={"sigma": sig}) for sig in sigma_range]
    return Ks, y


def process():
    """ Run experiments for all Keel datasets. """
    for dataset, func in it.product(KEEL_DATASETS, functions):
        # Replicates given a cost function
        Ps = np.zeros((replicates, rank))
        try:
            for repl in range(replicates):
                Ks, y = load_data_sample(dataset, Nmax=N, nk=nk)
                model = LarsMKL(rank=rank, delta=delta, f=func)
                model.fit(Ks, y)
                Ps[repl] = model.P
        except ValueError as ve:
            print(ve)
            continue

        # Plot
        fname = os.path.join(out_dir, "prog_%s_dataset_%s.pdf" % (func.__name__, dataset))
        plot_progressions(Ps, nk, title="%s | %s | N=%d" % (func.__name__, dataset, Ks[0].shape[0]))
        plt.savefig(fname)
        plt.close()
        print("Written %s" % fname)


if __name__ == "__main__":
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Makedir %s" % out_dir)
    process()

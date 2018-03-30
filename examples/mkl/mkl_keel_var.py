import scipy.stats as st
import numpy as np
import os
import csv
os.environ["OCTAVE_EXECUTABLE"] = "/usr/local/bin/octave"

# Kernels
from scipy.spatial.distance import cdist
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface

# Datasets
from datasets.keel import load_keel, KEEL_DATASETS

# Utils
import matplotlib.pyplot as plt

# New methods
from examples.mkl.mkl_est_var import estimate_variance_cv, plot_variance_cv

hlp = """
    Estimate the variance and lengthscale for KEEL regression datasets. 
"""

# Paths
out_dir = "/Users/martins/Dev/mklaren/examples/mkl/output/mkl_keel_var"
res_dir = "/Users/martins/Dev/mklaren/examples/mkl/results/mkl_keel_var"
N = 1000


# TODO: add to general tools
def estimate_sigma_dist(X, q=10):
    """ Estimate distribution of lengthscales in the dataset.
        Return lengthscales such that they cover the distribution of distances in the dataset
        (using q percentiles).
    """
    n = X.shape[0]
    D = cdist(X, X) * np.tri(n)
    d = D[np.where(D)]
    return np.percentile(d, q=np.linspace(0, 100, q))


def process(dataset):
    """ Generate results for a dataset and return csv row. """

    # Load data
    data = load_keel(n=N, name=dataset)
    inxs = np.argsort(data["target"])
    X = st.zscore(data["data"], axis=0)[inxs]
    y = st.zscore(data["target"])[inxs]

    # Estimate sigma distribution and range of kernels
    sigma_range = estimate_sigma_dist(X)
    Ks = [Kinterface(data=X,
                     kernel=exponential_kernel,
                     kernel_args={"sigma": sig})
             for sig in sigma_range]

    # Estimate variance based on CV
    try:
        S_tr, S_te, var_est = estimate_variance_cv(Ks, y, lbd_min=-7, lbd_max=5, n_lbd=20)
    except ValueError as ve:
        print("Variance estimation error; dataset %s " % dataset)
        print(ve)
        return None

    # Estimate SNR
    f_var = np.var(y) - var_est
    snr = f_var / var_est


    # Variance vs. sigma estimate
    fname = os.path.join(out_dir, "var_%s_lin.pdf" % dataset)
    plot_variance_cv(S_tr, S_te, log=False)
    plt.title(dataset)
    plt.savefig(fname)
    plt.close()
    print("Written %s" % fname)

    # Variance vs. sigma estimate
    fname = os.path.join(out_dir, "logvar_%s_log.pdf" % dataset)
    plot_variance_cv(S_tr, S_te, log=True)
    plt.title(dataset)
    plt.savefig(fname)
    plt.close()
    print("Written %s" % fname)

    # Draw length scales
    fname = os.path.join(out_dir, "scales_%s.pdf" % dataset)
    fig, axes = plt.subplots(nrows=len(Ks), ncols=1, figsize=(5, 10))
    for i, ax in enumerate(axes):
        ax.plot(Ks[i][:, len(X) / 2])
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel("Input space")
    axes[0].set_title("%s $k(x_{n/2}, \\cdot)$" % dataset)
    fig.tight_layout()
    plt.savefig(fname)
    plt.close()
    print("Written %s" % fname)

    # Store results
    return {"dataset": dataset,
            "N": len(X),
            "var": np.round(var_est, 5),
            "snr": np.round(snr, 5)}


if __name__ == "__main__":

    # Mkdir
    for d in (res_dir, out_dir):
        if not os.path.exists(d):
            os.makedirs(d)
            print("Makedir %s" % d)

    # Open output stream
    header = ["replicate", "dataset", "N", "var", "snr"]
    fname = os.path.join(res_dir, "results.csv")
    fp = open(fname, "w", buffering=0)
    writer = csv.DictWriter(fp, fieldnames=header)
    writer.writeheader()
    count = 0
    replicates = 10

    # Process
    for repl in range(replicates):
        for dset in KEEL_DATASETS:
            try:
                row = process(dset)
                if row is not None:
                    count += 1
                    row["replicate"] = repl
                    writer.writerow(row)
                    print("Written %d rows to %s" % (count, fname))
            except Exception as e:
                print("Exception with %s: %s" % (dset, e.message))

    fp.close()
    print("End")

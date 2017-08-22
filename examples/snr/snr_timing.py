hlp = """
    Timing experiments spawning multiple processes to limit the execution time.
    CSI does not work due to unknown subprocessing (oct2py) issues.
"""
import os
import csv
import time
import datetime
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import pickle
import shutil
import subprocess
from examples.snr.snr import generate_data, test, meth2color
from multiprocessing import Manager, Process
from mklaren.regression.ridge import RidgeMKL


# Global method list
METHODS = list(RidgeMKL.mkls.keys()) + ["Mklaren", "ICD", "Nystrom", "RFF", "FITC", "CSI"]
# METHODS = ["CSI"]
# PYTHON = "/Users/martin/Dev/py2/bin/python"

# Run CSI only
TMP_DIR = "temp"
PYTHON = "python"
SCRIPT = "snr_timing_child.py"

def wrapsf(Ksum, Klist, inxs, X, Xp, y, f, method, return_dict):
    """ Worker thread ; compute method running time;
        funky behaviour on OSX if you run a simple np.linalg.inv ; works on Ubuntu; """
    r = test(Ksum, Klist, inxs, X, Xp, y, f, methods=(method,), lbd=0.1)
    return_dict[method] = r[method]["time"]
    return

def wrapCSI(Ksum, Klist, inxs, X, Xp, y, f, method, return_dict):
    """ Wrap CSI in an outside process"""
    obj = (Ksum, Klist, inxs, X, Xp, y, f)
    fname = os.path.join(TMP_DIR, "%s.in.pkl" % hash(str(obj)))
    fout = os.path.join(TMP_DIR, "%s.out.pkl" % hash(str(obj)))
    pickle.dump(obj, open(fname, "w"), protocol=pickle.HIGHEST_PROTOCOL)
    subprocess.call([PYTHON, SCRIPT, fname, fout])
    r = pickle.load(open(fout))
    t = r[method]["time"]
    return_dict[method] = t
    return

def cleanup():
    """ Cleanup after CSI subprocess if killed. """
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)
    return


def process():
    # Fixed hyperparameters
    n_range = np.logspace(2, 6, 9).astype(int)
    rank_range = [5, 10, 30]
    p_range = [1, 3, 10]
    d_range = [1, 10, 100]
    limit = 3600 # 60 minutes

    # Safe guard dict to kill off the methods that go over the limit
    # Set a prior limit to full-rank methods to 4e5
    off_limits = dict([(m, int(4e5)) for m in RidgeMKL.mkls.keys()])

    # Fixed output
    # Create output directory
    d = datetime.datetime.now()
    dname = os.path.join("..", "output", "snr", "timings",
                         "%d-%d-%d" % (d.year, d.month, d.day))
    if not os.path.exists(dname): os.makedirs(dname)
    rcnt = len(os.listdir(dname))
    fname = os.path.join(dname, "results_%d.csv" % rcnt)
    print("Writing to %s ..." % fname)

    # Output
    header = ["kernel", "d", "n", "p", "method", "lambda", "rank", "limit", "time"]
    fp = open(fname, "w", buffering=0)
    writer = csv.DictWriter(fp, fieldnames=header)
    writer.writeheader()

    # Main loop
    for input_dim, P, rank, n in it.product(d_range, p_range, rank_range, n_range):

        # Cleanup
        cleanup()

        # Generate a dataset of give rank
        gamma_range = np.logspace(-3, 6, P)
        Ksum, Klist, inxs, X, Xp, y, f = generate_data(n=n,
                                                       rank=rank,
                                                       inducing_mode="uniform",
                                                       noise=1.0,
                                                       gamma_range=gamma_range,
                                                       input_dim=input_dim,
                                                       signal_sampling="weights",
                                                       data="random")
        # Print after dataset generation
        dat = datetime.datetime.now()
        print("%s\td=%d n=%d rank=%d p=%d" % (dat, input_dim, n, rank, P))

        # Evaluate methods
        manager = Manager()
        return_dict = manager.dict()
        jobs = dict()
        for method in METHODS:
            if off_limits.get(method, np.inf) <= n:
                print("%s is off limit for d=%d n=%d rank=%d p=%d" % (method, input_dim, n, rank, P))
                return_dict[method] = float("inf")
                continue
            if method == "CSI":
                p = Process(target=wrapCSI, name="test_%s" % method,
                            args=(Ksum, Klist, inxs, X, Xp,
                                  y, f, method, return_dict))
            else:
                p = Process(target=wrapsf, name="test_%s" % method,
                            args=(Ksum, Klist, inxs, X, Xp,
                                  y, f, method, return_dict))
            p.start()
            jobs[method] = p

        # Kill jobs exceeding time limit
        time_start = time.time()
        while True:
            time.sleep(1)
            alive = any([p.is_alive() for p in jobs.values()])
            if not alive:
                break
            t = time.time() - time_start
            if t > limit:
                for method, p in jobs.items():
                    if p.is_alive():
                        # Terminate process and store method to off limits for this n
                        # Note that this is the minimal point in (n, p, rank) for which if doesn't work
                        print("%s REGISTERED for d=%d n=%d rank=%d p=%d" % (method, input_dim, n, rank, P))
                        return_dict[method] = float("inf")
                        off_limits[method] = min(off_limits.get(method, np.inf), n)
                        p.terminate()

        # Write to output
        for method, value in return_dict.items():
            row = {"kernel": Klist[0].kernel.__name__,
                    "d": input_dim, "n": n, "p": P, "method": method, "limit": limit,
                   "lambda": 0.1, "rank": rank, "time": value}
            writer.writerow(row)


def plot_timings(fname, num_kernels=10, dims=(1, 10, 100)):
    """
    Summary plot of timings.
    :param fname: Results.csv file
    :param num_kernels: Number of kernels.
    :param dims: Selected dimensions.
    :return:
    """
    # Output
    out_dir = "/Users/martin/Dev/mklaren/examples/output/snr/timings/"

    # Read header and data
    cols = list(np.genfromtxt(fname, delimiter=",", dtype="str", max_rows=1))
    data = np.genfromtxt(fname, delimiter=",", dtype="str", skip_header=1)

    # Filter by number of kernels.
    num_k = np.array(data[:, cols.index("p")]).astype(int)
    inxs = num_k == num_kernels

    # Read columns
    method = np.array(data[inxs, cols.index("method")]).astype(str)
    n = np.array(data[inxs, cols.index("n")]).astype(int)
    dim = np.array(data[inxs, cols.index("d")]).astype(int)
    rank = np.array(data[inxs, cols.index("rank")]).astype(int)
    tm = np.array(data[inxs, cols.index("time")]).astype(float)
    mins = tm / 60.0

    # Set figure
    ranks = sorted(set(rank))
    fig, axes = plt.subplots(figsize=(3*3.5, 2*3.5),
                           ncols=len(ranks), nrows=len(dims),
                           sharex=True, sharey=True)
    for d, r in it.product(dims, ranks):
        i, j = dims.index(d), ranks.index(r)
        ax = axes[i][j]
        for meth in sorted(set(method), key=lambda m: m not in RidgeMKL.mkls.keys()):
            fmt = "s--" if meth in RidgeMKL.mkls.keys() else "s-"
            inxs = ((rank == r) * (dim == d) * (method == meth)).astype(bool)
            ax.plot(np.log10(n[inxs]), np.log10(mins[inxs]), fmt, label=meth,
                     linewidth=2, color=meth2color[meth])
        ax.grid("on")
        if j == 0: ax.set_ylabel("log10 time (mins)")
        if i == len(dims) - 1: ax.set_xlabel("log10 n")
        ax.set_title("Rank: %d, input dim: %d" % (r, d))
    axes[0][0].legend(ncol=len(set(method))/2, loc=(0, 1.3), frameon=False)
    pdfile = os.path.join(out_dir, "timings_p-%d.pdf" % num_kernels)
    epsfile = os.path.join(out_dir, "timings_p-%d.eps" % num_kernels)
    plt.savefig(pdfile, bbox_inches="tight")
    plt.savefig(epsfile, bbox_inches="tight")
    print("Written %s" % pdfile)
    print("Written %s" % epsfile)
    plt.close()


if __name__ == "__main__":
    process()
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
from examples.snr.snr import generate_data, test, meth2color
from multiprocessing import Manager, Process
from mklaren.regression.ridge import RidgeMKL


# Global method list
METHODS = list(RidgeMKL.mkls.keys()) + ["Mklaren", "ICD", "Nystrom", "RFF", "FITC"]

def wrapsf(Ksum, Klist, inxs, X, Xp, y, f, method, return_dict):
    """ Worker thread ; compute method running time;
        funky behaviour on OSX if you run a simple np.linalg.inv ; works on Ubuntu; """
    r = test(Ksum, Klist, inxs, X, Xp, y, f, methods=(method,), lbd=0.1)
    return_dict[method] = r[method]["time"]
    return

def process():
    """ Run main loop. """

    # Fixed hyperparameters
    n_range = np.logspace(2, 6, 9).astype(int)
    rank_range = [5, 10, 30]
    p_range = [1, 3, 10]
    d_range = [1, 3, 10, 30, 100]
    limit = 3600 # 60 minutes

    # Safe guard dict to kill off the methods that go over the limit
    # Set a prior limit to full-rank methods to 1e5
    off_limits = dict([(m, int(2e5)) for m in RidgeMKL.mkls.keys()])

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
            p = Process(target=wrapsf, name="test",
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


def plot_timings(fname):
    """
    Summary plot of timings.
    :param fname: Results.csv file
    :return:
    """
    # Output
    out_dir = "/Users/martin/Dev/mklaren/examples/output/snr/timings/"

    # Read header and data
    cols = list(np.genfromtxt(fname, delimiter=",", dtype="str", max_rows=1))
    data = np.genfromtxt(fname, delimiter=",", dtype="str", skip_header=1)

    # Read columns
    n = np.array(data[:,cols.index("n")]).astype(int)
    rank = np.array(data[:, cols.index("rank")]).astype(int)
    num_k = np.array(data[:, cols.index("p")]).astype(int)
    method = np.array(data[:, cols.index("method")]).astype(str)

    # Time (minutes)
    tm = np.array(data[:, cols.index("time")]).astype(float)
    mins = tm / 60.0

    # Set figure
    ps = sorted(set(num_k))
    ranks = sorted(set(rank))
    fig, axes = plt.subplots(figsize=(3*3.5, 2*3.5),
                           ncols=len(ranks), nrows=len(ps),
                           sharex=True, sharey=True)
    for p, r in it.product(ps, ranks):
        i, j = ps.index(p), ranks.index(r)
        ax = axes[i][j]
        for meth in sorted(set(method), key=lambda m: m not in RidgeMKL.mkls.keys()):
            fmt = "s--" if meth in RidgeMKL.mkls.keys() else "s-"
            inxs = ((rank == r) * (p == num_k) * (method == meth)).astype(bool)
            ax.plot(np.log10(n[inxs]), np.log10(mins[inxs]), fmt, label=meth,
                     linewidth=2, color=meth2color[meth])
        ax.grid("on")
        if j == 0: ax.set_ylabel("log10 time (mins)")
        if i == len(ps) - 1: ax.set_xlabel("log10 n")
        ax.set_title("Rank: %d, num. kernels: %d" % (r, p))
    axes[0][0].legend(ncol=len(set(method))/2, loc=(0, 1.3))
    pdfile = os.path.join(out_dir, "timings.pdf")
    epsfile = os.path.join(out_dir, "timings.eps")
    plt.savefig(pdfile, bbox_inches="tight")
    plt.savefig(epsfile, bbox_inches="tight")
    print("Written %s" % pdfile)
    print("Written %s" % epsfile)
    plt.close()


if __name__ == "__main__":
    process()
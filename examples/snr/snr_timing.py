# -*- coding: utf-8 -*-
hlp = """
    Timing experiments spawning multiple processes to limit the execution time.
    CSI does not work due to unknown subprocessing (oct2py) issues.
"""
import sys
reload(sys)
sys.setdefaultencoding('utf8')

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

names = {"align": "Align",
         "alignf": "AlignF",
         "alignfc": "AlignFC",
         "Nystrom": "Nyström",
         "l2krr": "L2-KRR",
         "uniform": "Uniform"}

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
    # n_range = [1000]
    rank_range = [5, 10, 30]
    # rank_range = [30]
    p_range = [1, 3, 10]
    # p_range = np.logspace(1, 4, 13).astype(int)
    d_range = [100]
    limit = 3600 # 60 minutes

    # Safe guard dict to kill off the methods that go over the limit
    # Set a prior limit to full-rank methods to 4e5
    off_limits = dict([(m, int(4e5)) for m in RidgeMKL.mkls.keys()])
    # off_limits = dict()

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
            # if off_limits.get(method, np.inf) <= P:
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
                        # off_limits[method] = min(off_limits.get(method, np.inf), P)
                        off_limits[method] = min(off_limits.get(method, np.inf), n)
                        p.terminate()

        # Write to output
        for method, value in return_dict.items():
            row = {"kernel": Klist[0].kernel.__name__,
                    "d": input_dim, "n": n, "p": P, "method": method, "limit": limit,
                   "lambda": 0.1, "rank": rank, "time": value}
            writer.writerow(row)


def plot_timings(fname, ranks=(5, 30), kernels=(1, 10)):
    """
    Summary plot of timings.
    :param fname: Results.csv file
    :param ranks: Selected ranks.
    :param kernels: Selected number of kernels.
    :return:
    """
    # Output
    out_dir = "/Users/martin/Dev/mklaren/examples/output/snr/timings/"

    # Read header and data
    cols = list(np.genfromtxt(fname, delimiter=",", dtype="str", max_rows=1))
    data = np.genfromtxt(fname, delimiter=",", dtype="str", skip_header=1)

    # Filter by number of kernels.
    num_k = np.array(data[:, cols.index("p")]).astype(int)

    # Read columns
    method = np.array(data[:, cols.index("method")]).astype(str)
    n = np.array(data[:, cols.index("n")]).astype(int)
    rank = np.array(data[:, cols.index("rank")]).astype(int)
    tm = np.array(data[:, cols.index("time")]).astype(float)
    mins = tm / 60.0

    # Set figure
    fig, axes = plt.subplots(figsize=(5, 4.5),
                           ncols=len(ranks), nrows=len(kernels),
                           sharex=True, sharey=True)
    for p, r in it.product(kernels, ranks):
        i, j = kernels.index(p), ranks.index(r)
        ax = axes[i][j]
        for meth in sorted(set(method), key=lambda m: (m not in RidgeMKL.mkls.keys(), m.lower())):
            fmt = "s--" if meth in RidgeMKL.mkls.keys() else "s-"
            inxs = ((rank == r) * (num_k == p) * (method == meth)).astype(bool)
            ax.plot(np.log10(n[inxs]), np.log10(mins[inxs]), fmt, label=names.get(meth, meth),
                     linewidth=2, color=meth2color[meth])
        ax.grid("on")
        if j == 0: ax.set_ylabel("log10 time (mins)")
        if i == len(kernels) - 1: ax.set_xlabel("log10 n")
        ax.set_title("Rank: %d, num. kernels: %d" % (r, p))
    axes[0][0].legend(ncol=int(np.ceil(len(set(method))/4.0)), loc=(0, 1.3), frameon=False)
    pdfile = os.path.join(out_dir, "timings.pdf")
    epsfile = os.path.join(out_dir, "timings.eps")
    plt.savefig(pdfile, bbox_inches="tight")
    plt.savefig(epsfile, bbox_inches="tight")
    print("Written %s" % pdfile)
    print("Written %s" % epsfile)
    plt.close()


if __name__ == "__main__":
    process()
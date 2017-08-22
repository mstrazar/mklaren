# -*- coding: utf-8 -*-
"""
Motivation: If one is allowed to sample columns with regularization,
this leads to lower errors at high noise levels.

Comparison of the three pivot selection methods with varying noise rates
on a simple Gaussian Process signal.

There is no training and test set, just comparison with recovering true
signal.

How to select lambda?
    - Display results for all lambda.

Add more kernels?

"""
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import sys
import csv
import datetime
import os
import itertools as it
import time
import scipy
import numpy as np
from scipy.stats import multivariate_normal as mvn, pearsonr, entropy
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from mklaren.regression.fitc import FITC
from mklaren.projection.rff import RFF
from mklaren.regression.ridge import RidgeMKL
import matplotlib.pyplot as plt
import pickle, gzip


# Method order if required
meth_order = ["Mklaren", "CSI", "ICD", "Nystrom", "RFF", "FITC", "True"]

# Color mappings
meth2color = {"Mklaren": "green",
              "CSI": "red",
              "ICD": "blue",
              "Nystrom": "pink",
              "FITC": "orange",
              "RFF": "magenta",
              "True": "black",
              "l2krr": "green",
              "align": "pink",
              "uniform": "blue",
              "alignf": "red",
              "alignfc": "orange"}


def generate_data(n, rank,
                  inducing_mode="uniform", noise=1, gamma_range=(0.1,), seed=None,
                  input_dim=1, signal_sampling="GP", data="mesh"):
    """
    Generate an artificial dataset with imput dimension.
    :param n: Number od data points.
    :param rank: Number of inducing points.
    :param inducing_mode:   Biased or uniform distribution of data points.
    :param noise: Noise variance.
    :param gamma_range: Number of kernels and hyperparameters.
    :param seed: Random seed.
    :param input_dim: Input space dimension.
    :param signal_sampling: 'GP' or 'weights'. Weights is more efficient.
    :param data: mesh or input_dim.
    :return:
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate data for arbitray input_dim
    if data == "mesh":
        x = np.linspace(-10, 10, n).reshape((n, 1))
        M = np.meshgrid(*(input_dim * [x]))
        X = np.array(zip(*[m.ravel() for m in M]))
        N = X.shape[0]

        xp = np.linspace(-10, 10, 100).reshape((100, 1))
        Mp = np.meshgrid(*(input_dim * [xp]))
        Xp = np.array(zip(*[m.ravel() for m in Mp]))
    elif data == "random":
        # Ensure data is separated at proper lengthscales
        ls = FITC.gamma2lengthscale(min(gamma_range)) / np.sqrt(input_dim)
        a, b = -n * ls / 2.0,  n * ls / 2.0
        X = a + 2 * b * np.random.rand(n, input_dim)
        N = X.shape[0]
        Xp = np.random.rand(100, input_dim)
    else:
        raise ValueError("Unknown data mode: %s" % data)

    # Kernel sum
    Ksum = Kinterface(data=X, kernel=kernel_sum,
                      kernel_args={
                          "kernels": [exponential_kernel] * len(gamma_range),
                          "kernels_args": [{"gamma": g} for g in gamma_range]})

    # Sum of kernels
    Klist = [Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": g})
             for g in gamma_range]

    a = np.arange(X.shape[0], dtype=int)
    if inducing_mode == "uniform":
        p = None
    elif inducing_mode == "biased":
        af = np.sum(X + abs(X.min(axis=0)), axis=1)
        p = (af ** 2 / (af ** 2).sum())
    else:
        raise ValueError(inducing_mode)

    inxs = np.random.choice(a, p=p, size=rank, replace=False)
    if signal_sampling == "GP":
        Kny = Ksum[:, inxs].dot(np.linalg.inv(Ksum[inxs, inxs])).dot(Ksum[inxs, :])
        f = mvn.rvs(mean=np.zeros((N,)), cov=Kny)
        y = mvn.rvs(mean=f, cov=noise * np.eye(N, N))
    elif signal_sampling == "weights":
        L = Ksum[:, inxs].dot(scipy.linalg.sqrtm(np.linalg.inv(Ksum[inxs, inxs])))
        w = mvn.rvs(mean=np.zeros(rank,), cov=np.eye(rank, rank)).ravel()
        f = L.dot(w)
        y = f + np.random.rand(n, 1).ravel() * noise
    else:
        raise ValueError(signal_sampling)

    return Ksum, Klist, inxs, X, Xp, y, f


def plot_signal(X, Xp, y, f, models=None, tit="", typ="plot_models", f_out = None):
    """
    Plot fitted signal.

    :param X: Sampling coordinates.
    :param Xp:  Plotting (whole signal) coordinates.
    :param y:   True observed values.
    :param f:   True signal.
    :param models: Onr dictionary per model;
        "yp"    Predicted signal at yp.
        "anchors" Anchor (inducing points coordinates), one set per lengthscale.
        "color": Color.
        "label": Name.
    :param tit:
    :param typ: plot_models or plot_gammas
    :return:
    """

    # Plot signal
    plt.figure()
    x = X.ravel()
    xp = Xp.ravel()
    xmin, xmax = xp.min(), xp.max()
    ymin, ymax = int(min(f.min(), y.min())) - 1, int(max(f.max(), y.max())) + 1

    # Plot data
    plt.plot(x, y, "k.")
    plt.plot(x, f, "r--")

    # Compute anchor ticks
    P = max([1] + map(lambda m: len(m.get("anchors", [])), models.values()))

    if typ == "plot_gammas":
        Gxs = [np.linspace(xmin, xmax, 5 + 10 * g) for g in np.logspace(-1, 1, P)]
    elif typ == "plot_models":
        Gxs = [np.linspace(xmin, xmax, 15) for g in np.logspace(-1, 1, len(models))]
    else:
        raise ValueError
    Gys = range(ymin - len(Gxs), ymin)

    # Plot freqency scales
    for gi, (gx, gy) in enumerate(zip(Gxs, Gys)):
        plt.plot(gx, [gy] * len(gx), "|", color="gray")

    # Plot multiple signals and anchors
    if models is not None:
        for mi, (label, data) in enumerate(models.items()):
            if label == "True": continue
            yp = data.get("yp", np.zeros((len(X), )))
            color = meth2color[label]
            plt.plot(xp, yp, "-", color=color, label="%s" % label)

    for mi, (label, data) in enumerate(sorted(models.items(), key=lambda lb: lb[0] == "True")):
            anchors = data.get("anchors", [[]])
            color = meth2color[label]
            if typ == "plot_gammas":        # Draw for different gammas
                for gi in range(P):
                    if len(anchors) <= gi or not len(anchors[gi]): continue
                    plt.plot(anchors[gi], [Gys[gi]] * len(anchors[gi]), "^",
                             color=color, markersize=8, alpha=0.6)

            elif typ == "plot_models":      # Draw for different methods
                gi = mi
                ancs = np.array(anchors).ravel()
                plt.text(xmin - 1, Gys[gi], "[%s]" % label, horizontalalignment="right",
                         verticalalignment="center", color=meth2color[label])
                plt.plot(ancs, [Gys[gi]] * len(ancs), "^",
                         color=color, markersize=8, alpha=0.6)

    plt.title(tit)
    plt.yticks(np.linspace(ymin, ymax, 2 * (ymax - ymin) + 1).astype(int))
    plt.ylim((ymin - len(Gys) - 1, ymax))
    plt.xlabel("Input space (x)")
    plt.ylabel("Output space (y)")
    plt.gca().yaxis.set_label_coords(-0.05, 0.75)

    if f_out is None:
        plt.show()
    else:
        plt.savefig(f_out)
        plt.close()
        print("Written %s" % f_out)


def plot_signal_subplots(X, Xp, y, f, models=None, f_out=None):
    """
    Plot fitted signal on multiple plots to avoid clutter.
    Models dictionary does not assume the 'True' model

    :param X: Sampling coordinates.
    :param Xp:  Plotting (whole signal) coordinates.
    :param y:   True observed values.
    :param f:   True signal.
    :param models: Onr dictionary per model;
        "yp"    Predicted signal at yp.
        "anchors" Anchor (inducing points coordinates), one set per lengthscale.
        "color": Color.
        "label": Name.
    :param f_out: Output file. If not provided, show plot on screen.
    :return:
    """
    x = X.ravel()
    xp = Xp.ravel()
    xmin, xmax = min(0, xp.min()), xp.max()
    ymin, ymax = y.min(), y.max()
    nmods = len(models)

    fig, ax = plt.subplots(sharex=True, ncols=1, nrows=nmods, figsize=(4.33, nmods * 0.8))
    for mi, (label, data) in enumerate(sorted(models.items(), key=lambda t: meth_order.index(t[0]))):
        lbl = label.replace("Nystrom", "Nyström")
        yp = data.get("yp", np.zeros((len(X),)))
        color = meth2color[label]

        # Plot to axis
        ax[mi].set_xlim(xmin, xmax)
        ax[mi].set_ylim(ymin, ymax)
        ax[mi].plot(x, y, ".", color="gray")
        if f is not None: ax[mi].plot(x, f, "r--")
        ax[mi].plot(xp, yp, "-", color=color, label="%s" % label, linewidth=1.5)

        # Plot anchors if provided
        anchors = data.get("anchors", [[]])
        ancs = np.array(anchors).ravel()
        ax[mi].plot(ancs, [ymin + (ymax - ymin) * 0.05] * len(ancs),
                    "^", color=color, markersize=8, alpha=0.6)
        ax[mi].set_ylabel(lbl)

    ax[-1].set_xlabel("Input space (x)")
    fig.tight_layout()

    if f_out is None:
        plt.show()
    else:
        plt.savefig(f_out)
        plt.close()
        print("Written %s" % f_out)

        f_out_gz = f_out + ".pkl.gz"
        obj = (X, Xp, y, f, models)
        pickle.dump(obj, gzip.open(f_out_gz, "w"), protocol=pickle.HIGHEST_PROTOCOL)
        print("Written %s" % f_out_gz)



def plot_signal_2d(X, Xp, y, f, models=None, tit=""):
    # Plot signal
    N = X.shape[0]
    n = int(N ** 0.5)
    Np = Xp.shape[0]
    nn = int(Np ** 0.5)

    F = f.reshape((n, n))
    Y = y.reshape((n, n))

    methods = set(models.keys()) - {"True"}
    fig, ax = plt.subplots(nrows=2, ncols=len(methods))
    ax[0][0].imshow(F)
    ax[0][0].set_title("True signal")
    ax[0][1].imshow(Y)
    ax[0][1].set_title("Signal + noise")
    for i in range(2, len(methods)):
        ax[0][i].axis("off")


    for mi, m in enumerate(methods):
        yp = models[m]["yp"]
        Yp = yp.reshape((nn, nn))
        ax[1][mi].imshow(Yp)
        ax[1][mi].set_title(m)

    plt.show()


def test(Ksum, Klist, inxs, X, Xp, y, f, delta=10, lbd=0.1, kappa=0.99,
         methods=("Mklaren", "ICD", "CSI", "Nystrom", "FITC")):
    """
    Sample data from a Gaussian process and compare fits with the sum of kernels
    versus list of kernels.
    :param Ksum:
    :param Klist:
    :param inxs:
    :param X:
    :param Xp:
    :param y:
    :param f:
    :param delta:
    :param lbd:
    :param methods:
    :return:
    """
    def flatten(l):
        return [item for sublist in l for item in sublist]

    P = len(Klist)           # Number of kernels
    rank = len(inxs)         # Total number of inducing points over all lengthscales
    anchors = X[inxs,]

    # True results
    results = {"True": {"anchors": anchors,
              "color": "black"}}

    # Fit MKL for kernel sum and
    if "Mklaren" in methods:
        mkl = Mklaren(rank=rank,
                      delta=delta, lbd=lbd)
        t1 = time.time()
        mkl.fit(Klist, y)
        t2 = time.time() - t1
        y_Klist = mkl.predict([X] * len(Klist))
        yp_Klist = mkl.predict([Xp] * len(Klist))
        active_Klist = [flatten([mkl.data.get(gi, {}).get("act", []) for gi in range(P)])]
        anchors_Klist = [X[ix] for ix in active_Klist]
        rho_Klist, _ = pearsonr(y_Klist, f)
        evar = (np.var(y) - np.var(y - y_Klist)) / np.var(y)
        results["Mklaren"] = {
                     "rho": rho_Klist,
                     "active": active_Klist,
                     "anchors": anchors_Klist,
                     "sol_path": mkl.sol_path,
                     "yp": yp_Klist,
                     "time": t2,
                     "evar": evar,
                     "model": mkl,
                     "color": meth2color["Mklaren"]}

    # Fit CSI
    if "CSI" in methods:
        csi = RidgeLowRank(rank=rank, lbd=lbd,
                           method="csi", method_init_args={"delta": delta, "kappa": kappa},)
        t1 = time.time()
        csi.fit([Ksum], y)
        t2 = time.time() - t1
        y_csi = csi.predict([X])
        yp_csi = csi.predict([Xp])
        active_csi = csi.active_set_
        anchors_csi = [X[ix] for ix in active_csi]
        try:
            rho_csi, _ = pearsonr(y_csi, f)
        except Exception as e:
            rho_csi = 0
        evar = (np.var(y) - np.var(y - y_csi)) / np.var(y)
        results["CSI"] = {
                "rho": rho_csi,
                "active": active_csi,
                "anchors": anchors_csi,
                "time": t2,
                "yp": yp_csi,
                "evar": evar,
                "model": csi,
                "color": meth2color["CSI"]}

    # Fit RFF
    if "RFF" in methods:
        gamma_range = map(lambda k: k.kernel_args["gamma"], Klist)
        rff = RFF(delta=delta, rank=rank, lbd=lbd, gamma_range=gamma_range)
        t1 = time.time()
        rff.fit(X, y)
        t2 = time.time() - t1
        y_rff = rff.predict(X)
        yp_rff = rff.predict(Xp)
        rho_rff, _ = pearsonr(y_rff, f)
        evar = (np.var(y) - np.var(y - y_rff)) / np.var(y)
        results["RFF"] = {
            "rho": rho_rff,
            # "active": active_rff,
            # "anchors": anchors_rff,
            "time": t2,
            "yp": yp_rff,
            "evar": evar,
            "model": rff,
            "color": meth2color["RFF"]}

    # Fit FITC
    if "FITC" in methods:
        fitc = FITC(rank=rank)
        t1 = time.time()
        fitc.fit(Klist, y, optimize=True, fix_kernel=False)
        t2 = time.time() - t1
        y_fitc = fitc.predict([X]).ravel()
        yp_fitc = fitc.predict([Xp]).ravel()
        try:
            rho_fitc, _ = pearsonr(np.round(y_fitc, 4), f)
        except Exception as e:
            sys.stderr.write("FITC exception: %s\n" % e)
            rho_fitc = 0
        evar = (np.var(y) - np.var(y - y_fitc)) / np.var(y)

        # Approximate closest active index to each inducing point
        anchors = fitc.anchors_
        actives = [[np.argmin(np.sum((a - X)**2, axis=1)) for a in anchors]]

        results["FITC"] = {
            "rho": rho_fitc,
            "active": actives,
            "anchors": anchors,
            "time": t2,
            "yp": yp_fitc,
            "evar": evar,
            "model": fitc,
            "color": meth2color["FITC"]}



    # Fit ICD
    if "ICD" in methods:
        icd = RidgeLowRank(rank=rank, lbd=lbd,
                           method="icd")
        t1 = time.time()
        icd.fit([Ksum], y)
        t2 = time.time() - t1
        y_icd = icd.predict([X])
        yp_icd = icd.predict([Xp])
        active_icd = icd.active_set_
        anchors_icd = [X[ix] for ix in active_icd]
        rho_icd, _ = pearsonr(y_icd, f)
        evar = (np.var(y) - np.var(y - y_icd)) / np.var(y)
        results["ICD"] = {"rho": rho_icd,
                "active": active_icd,
                "anchors": anchors_icd,
                "yp": yp_icd,
                "time": t2,
                "evar": evar,
                "model": icd,
                "color": meth2color["ICD"]}

    # Fit Nystrom
    if "Nystrom" in methods:
        nystrom = RidgeLowRank(rank=rank,
                               lbd=lbd,
                               method="nystrom",
                               method_init_args={"lbd": lbd, "verbose": False})
        t1 = time.time()
        nystrom.fit([Ksum], y)
        t2 = time.time() - t1
        y_nystrom = nystrom.predict([X])
        yp_nystrom = nystrom.predict([Xp])
        active_nystrom = nystrom.active_set_
        anchors_nystrom = [X[ix] for ix in active_nystrom]
        rho_nystrom, _ = pearsonr(y_nystrom, f)
        evar = (np.var(y) - np.var(y - y_nystrom)) / np.var(y)
        results["Nystrom"] = {
                "rho": rho_nystrom,
                "active": active_nystrom,
                "anchors": anchors_nystrom,
                "yp": yp_nystrom,
                "time": t2,
                "evar": evar,
                "model": nystrom,
                "color": meth2color["Nystrom"]}

    # Fit MKL methods (just for time comparison)
    for method in set(RidgeMKL.mkls.keys()) & set(methods):
        model = RidgeMKL(lbd=lbd, method=method)
        t1 = time.time()
        model.fit(Klist, y)
        t2 = time.time() - t1
        results[method] = {"time": t2}

    return results


def hist_total_variation(h1, h2):
    """
    Total variation between two histograms, assuming identical bins.
    :param h1: Histogram.
    :param h2: Histogram.
    :return: Total variation distance.
    """
    return np.sum(np.absolute(h1 - h2))


def inducing_points_distance(A, B):
    """
    Compute distance betwen two sets of inducing points)
    :param A: Set of inducing points (in any dimension), given as coordinates.
        Inducing points can be defined in terms of the training set or pseudo-inducing points.
    :param B: Same as A.
    :return: Minimal distance given permutations of x.
    """
    A = np.array(A)
    B = np.array(B)
    inxs = range(len(B))
    return min((np.linalg.norm(A - B[list(ix)]) for ix in it.permutations(inxs)))

def bin_centers(bins):
    """
    Centers of histogram bins to plothistograms as lines
    :param bins:
        Bins limits.
    :return:
        Centers of bins.
    """
    return bins[:-1] + (bins[1:] - bins[:-1])  / 2.0


def generate_noise(n, noise_model, input_dim):
    """
    Generate noise vector.
    :param n: Number of data samples along a dimension.
    :param noise_model: "fixed" or "increasing".
    :param input_dim:  Input dimensionality.
    :return:noise vector of size N = n ** input_dim.
    """
    N = n ** input_dim
    if noise_model == "fixed":
        if input_dim == 1:
            noise = 1
        elif input_dim == 2:
            noise = 0.03
    else:
        if input_dim == 1:
            noise = np.logspace(-2, 2, N)
        elif input_dim == 2:
            a = np.logspace(-1, 1, n).reshape((n, 1))
            A = a.dot(a.T) * 0.01
            noise = A.reshape((N, 1))
    return noise


def split_results(in_file, out_dir):
    """
    One-time use function to split results.pkl into a file readable by R.
    :param in_file: Input pkl.gz file.
    :param out_dir: Output directory.
    :return:
    """
    data = pickle.load(gzip.open(in_file))
    for row in data:
        fname = "actives_method-%s_noise-%s_sampling-%s_n-%d_rank-%d_lbd-%.3f_gamma-%.3f.txt" % \
                (row["method"], row["noise.model"], row["sampling.model"],
                 row["n"], row["rank"], row.get("lbd", 0), row["gamma"])
        actives = np.array(row["avg.actives"], dtype=int)
        np.savetxt(os.path.join(out_dir, fname), actives, fmt="%d")
    print("Saved %d files" % len(data))
    return


def cumulative_histogram(in_file, out_dir):
    """
    :param in_file: Input pkl.gz file.
    :param out_dir: Output directory.
    :return:
    """
    data = pickle.load(gzip.open(in_file))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Organize by hyperparameters
    hists = dict()
    for row in data:
        ky0 = (row["n"], row["rank"], row.get("lbd", 0), row["gamma"])
        ky1 = (row["noise.model"], row["sampling.model"])
        method = row["method"]
        actives = np.array(row["avg.actives"], dtype=int)
        if ky0 not in hists: hists[ky0] = dict()
        if ky1 not in hists[ky0]: hists[ky0][ky1] = dict()
        hists[ky0][ky1][method] = actives

    # Composite figure
    cols = ["fixed", "increasing"]
    rows = ["uniform", "biased"]
    for ky0, dd in hists.items():
        n, rank, lbd, gamma = ky0
        fig, axes = plt.subplots(figsize=(4.72, 4.72), ncols=2, nrows=2, sharex=True, sharey=True)
        for (noise, sampling), samples in dd.iteritems():
            i, j = rows.index(sampling), cols.index(noise)
            ax = axes[i][j]
            for meth, samp in samples.iteritems():
                counts, bins = np.histogram(samp, normed=False)
                probs = 1.0 * counts / counts.sum()
                centers = bin_centers(bins)
                cums = np.cumsum(probs)
                fmt = "--" if meth == "True" else "s-"
                label = meth.replace("Nystrom", "Nyström")
                if i + j == 0: ax.plot(centers, cums, fmt, color=meth2color[meth], linewidth=2, label=label)
                else:          ax.plot(centers, cums, fmt, color=meth2color[meth], linewidth=2)
                if j == 0: ax.set_ylabel("Cumulative probability")
                if i == 1: ax.set_xlabel("Inducing point (pivot) location")
                ax.set_title("Noise: %s \n sampling: %s" % (noise, sampling))
                ax.grid("on")

        # Layout + legend
        axes[0][0].legend(ncol=3, loc=(0, 1.3), frameon=False)
        fig.tight_layout()

        # Store to files
        for ext in ("pdf", "eps"):
            fname = "cumhist_n-%d_rank-%d_lbd-%.3f_gamma-%.3f.%s" % (n, rank, lbd, gamma, ext)
            plt.savefig(os.path.join(out_dir, fname), bbox_inches="tight")
            print("Written %s" % fname)
        plt.close()



def example_models(out_dir):
    """
    Example model fit to generate a figure.
    :return:
    """
    n = 100
    for noise_model, inducing_model, seed in it.product(("fixed", "increasing"), ("uniform", "biased"), range(0, 3)):
        fname = os.path.join(out_dir, "example_%s_%s_%d.pdf" % (noise_model, inducing_model, seed))
        noise = generate_noise(n, noise_model, 1)

        Ksum, Klist, inxs, X, Xp, y, f = generate_data(n=n,
                                                       rank=5,
                                                       inducing_mode=inducing_model,
                                                       noise=noise,
                                                       gamma_range=[1.0],
                                                       seed=seed,
                                                       input_dim=1)

        # Evaluate methods
        r = test(Ksum, Klist, inxs, X, Xp, y, f)
        plot_signal(X, Xp, y, f, models=r, tit="", f_out=fname)

    return


def generate_GP_samples():
    """
    One-time function to demonstrate samples from degenerate GPs with a sampling mode.
    :return:
    """
    noise = np.logspace(-2, 2, 100)
    Ksum, Klist, inxs, X, Xp, y, f = generate_data(n=100,
                                                   rank=3,
                                                   inducing_mode="biased",
                                                   noise=noise,
                                                   gamma_range=[0.3],
                                                   seed=None,
                                                   input_dim=1)

    plt.figure()
    plt.plot(f, label="signal")
    plt.plot(y, "k.", label="data")
    plt.xlabel("Input space (1D)")
    plt.ylabel("y")
    plt.show()


def main():
    # Experiment paramaters
    n_range = (100, )
    input_dim = 1

    rank_range = (3, 5,)
    lbd_range = (0, )
    gamma_range = [0.1, 0.3, 1, 3]
    repeats = 500
    pc = 0.1 # pseudocount; prevents inf in KL-divergence.
    noise_models = ("fixed", "increasing")
    sampling_models = ("uniform", "biased")
    methods = ("Mklaren", "CSI", "ICD", "Nystrom", "FITC")

    # Create output directory
    d = datetime.datetime.now()
    dname = os.path.join("..", "output", "snr", "%d-%d-%d" % (d.year, d.month, d.day))
    if not os.path.exists(dname):
        os.makedirs(dname)
    rcnt = len(os.listdir(dname))
    subdname = os.path.join(dname, "details_%d" % rcnt)
    if not os.path.exists(subdname):
        os.makedirs(subdname)
    fname = os.path.join(dname, "results_%d.csv" % rcnt)
    fname_details = os.path.join(subdname, "results.pkl.gz")
    print("Writing to %s ..." % fname)

    # Output file
    header = ["method", "noise.model", "sampling.model", "n", "rank", "lbd", "gamma",
              "total.variation", "kl.divergence"]
    writer = csv.DictWriter(open(fname, "w", buffering=0), fieldnames=header)
    writer.writeheader()
    results = []

    count = 0
    for rank, lbd, gamma, n, noise_model, inducing_mode in it.product(rank_range,
                                                                     lbd_range,
                                                                     gamma_range,
                                                                     n_range,
                                                                     noise_models,
                                                                     sampling_models,):

        noise = generate_noise(n, noise_model, input_dim)

        avg_actives = dict()
        avg_anchors = dict()

        r = None
        for seed in range(repeats):
            # Generate data
            Ksum, Klist, inxs, X, Xp, y, f = generate_data(n=n,
                                                           rank=rank,
                                                           inducing_mode=inducing_mode,
                                                           noise=noise,
                                                           gamma_range=[gamma],
                                                           seed=seed,
                                                           input_dim=input_dim)
            # Evaluate methods
            try:
                r = test(Ksum, Klist, inxs, X, Xp, y, f, methods=methods)
                # plot_signal(X, Xp, y, f, models=r, tit="")
                # plot_signal_2d(X, Xp, y, f, models=r, tit="")
            except Exception as e:
                print("Exception, continuing: %s" % str(e))
                continue

            # Fill in anchors and active points
            avg_actives["True"] = avg_actives.get("True", []) + list(inxs)
            avg_anchors["True"] = avg_anchors.get("True", []) + list(r["True"]["anchors"])
            for m in methods:
                avg_actives[m] = avg_actives.get(m, []) + list(r[m]["active"][0])
                avg_anchors[m] = avg_anchors.get(m, []) + list(r[m]["anchors"][0])

        # Compare distributions
        bins = None
        if input_dim == 1:
            probs, bins = np.histogram(avg_anchors["True"], normed=False)
            probs = 1.0 * (probs + pc) / (probs + pc).sum()
        elif input_dim == 2:
            A = np.array(avg_anchors["True"])
            probs, b1, b2 = np.histogram2d(A[:, 0], A[:, 1], normed=False, bins=5)
            probs = probs.ravel()
            probs = 1.0 * (probs + pc) / (probs + pc).sum()
            bins = (b1, b2)

        rows = []
        for m in methods:
            if input_dim == 1:
                h = np.histogram(avg_anchors[m], normed=False, bins=bins)
            elif input_dim == 2:
                A = np.array(avg_anchors[m])
                h = np.histogram2d(A[:, 0], A[:, 1], normed=False, bins=bins)
            query = h[0].ravel()
            query = 1.0 * (query + pc ) / (query + pc).sum()
            kl = entropy(probs, query)
            tv = hist_total_variation(probs, query)
            row  = {"method": m, "noise.model": noise_model, "sampling.model": inducing_mode,
                    "n": n, "rank":rank, "lbd": lbd, "gamma": gamma,
                    "total.variation": tv, "kl.divergence": kl}
            rows.append(row)

            # Extended row for details
            row_extd = row.copy()
            row_extd["avg.anchors"] = avg_anchors[m]
            row_extd["avg.actives"] = avg_actives[m]
            results.append(row_extd)

        # True row
        row_true = {"method": "True", "noise.model": noise_model, "n": n, "rank": rank,
                    "sampling.model": inducing_mode, "gamma": gamma,
                    "avg.anchors": avg_anchors["True"], "avg.actives": avg_actives["True"]}
        results.append(row_true)

        # Write results
        writer.writerows(rows)
        pickle.dump(results, gzip.open(fname_details, "w"), protocol=pickle.HIGHEST_PROTOCOL)
        count += len(rows)
        print("%s Written %d rows"% (str(datetime.datetime.now()), count))

        if input_dim == 1:
            # Plot histograms
            figname = os.path.join(subdname, "hist_%s_%s_n-%d_rank-%d_lbd-%.3f_gamma-%.3f.pdf"
                                 % (noise_model, inducing_mode, n, rank, lbd, gamma))
            fig, ax = plt.subplots(nrows=len(methods)+1, ncols=1)
            ax[0].hist(avg_actives["True"], color="gray", label="True", bins=bins)
            for mi, m in enumerate(methods):
                ax[mi+1].hist(avg_actives[m], color=r[m]["color"], label=m, bins=bins)
            ax[len(methods)].set_xlabel("Inducing point index")
            for i in range(len(ax)):
                ax[i].legend()
                ax[i].set_xlim((0, n))
            fig.tight_layout()
            plt.savefig(figname)
            plt.close()
            print("Written %s" % figname)

            # Plot lines
            figname = os.path.join(subdname, "lines_%s_%s_n-%d_rank-%d_lbd-%.3f_gamma-%.3f.pdf"
                                   % (noise_model, inducing_mode, n, rank, lbd, gamma))

            centers = bin_centers(bins)
            plt.figure()
            for m in ["True"] + list(methods):
                p, _= np.histogram(avg_actives[m], bins=bins)
                q = (1.0 * p) / sum(p)
                plt.plot(centers, q, ("." if m != "True" else "") + "-",
                         color=r[m]["color"], label=m)
            plt.legend(loc=2)
            plt.xlabel("Incuding point index")
            plt.ylabel("Probability")
            plt.savefig(figname)
            plt.close()
            print("Written %s" % figname)


if __name__ == "__main__":
    main()
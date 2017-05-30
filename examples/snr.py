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
import matplotlib
matplotlib.use("Agg")

import csv
import datetime
import os
import itertools as it

import numpy as np
from scipy.stats import multivariate_normal as mvn, pearsonr, entropy
from mklaren.kernel.kernel import exponential_kernel, kernel_sum
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import pickle, gzip


def process():
    # Parameters
    gamma_range = 2 ** (np.linspace(-2, 2, 5))  # Arbitrary kernel hyperparameters
    delta = 10  # Arbitrary look-ahead parameter

    # Objective experimentation;
    n_range = [100, 300, 500]  # Vaste enough range of dataset sizes (which are full-rank)
    noise_range = np.logspace(-2, 2, 7)  # Range of noise levels
    repeats = range(10)  # Number of repeats
    rank_percents = [0.05, 0.1, 0.15]  # Rank percentages given n
    methods = ["Mklaren", "CSI", "ICD"]
    lbd_range = [0, 0.1, 0.3, 1, 3, 10, 30,
                 100]  # Vast enough range, such that methods should be able to capture optimum somewhere

    # Create output directory
    d = datetime.datetime.now()
    dname = os.path.join("output", "snr", "%d-%d-%d" % (d.year, d.month, d.day))
    if not os.path.exists(dname):
        os.makedirs(dname)
    fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
    print("Writing to %s ..." % fname)

    header = ["repl", "method", "n", "gamma", "lbd", "snr", "rank", "noise", "mse_sig", "mse_rel", "pr_rho", "pr_pval"]
    writer = csv.DictWriter(open(fname, "w", buffering=0),
                            fieldnames=header, quoting=csv.QUOTE_ALL)
    writer.writeheader()


    count = 0
    for repl, gamma, n, noise, lbd, rp in it.product(repeats, gamma_range, n_range,
                                                     noise_range, lbd_range, rank_percents):
        rank = max(5, int(rp * n))

        # Generate data
        X = np.linspace(-10, 10, n).reshape((n, 1))
        K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": gamma})

        f = mvn.rvs(mean=np.zeros((n, )), cov=K[:, :])
        y = mvn.rvs(mean=f, cov=noise * np.eye(n, n))

        # Mklaren
        yp = None
        rows = []
        for method in methods:
            # The difference is only in including lambda at the time of pivot selection
            try:
                if method == "Mklaren":
                    mkl = Mklaren(rank=rank,
                                  delta=delta, lbd=lbd)
                    mkl.fit([K], y)
                    yp = mkl.predict([X])

                elif method == "ICD":
                    icd = RidgeLowRank(rank=rank, lbd=lbd,
                                       method="icd")
                    icd.fit([K], y)
                    yp = icd.predict([X])

                elif method == "CSI":
                    icd = RidgeLowRank(rank=rank, lbd=lbd,
                                       method="csi", method_init_args={"delta": delta})
                    icd.fit([K], y)
                    yp = icd.predict([X])
            except Exception as e:
                print("%s exception: %s" % (method,  e.message))
                continue

            # Metrics
            mse_sig = mse(yp, f)
            mse_rel = mse(yp, f) / np.var(y)
            snr = np.var(f) / noise
            pr_rho, pr_pval = pearsonr(yp, f)

            row = {"repl": repl, "method": method, "n": n, "snr": snr, "lbd": lbd,
                   "rank": rank, "noise": np.round(np.log10(noise), 2), "mse_sig": mse_sig,
                   "mse_rel": mse_rel, "pr_rho": pr_rho, "pr_pval": pr_pval, "gamma": gamma}
            rows.append(row)

        if len(rows) == len(methods):
            writer.writerows(rows)
            count += len(rows)
            print("%s Written %d rows (n=%d)" % (str(datetime.datetime.now()), count, n))



def generate_data(n, rank, inducing_mode="uniform", noise=1, gamma_range=(0.1,), seed=None):
    """

    :param n: Number od data points.
    :param rank: Number of inducing points.
    :param inducing_mode:   Biased or uniform distribution of data points.
    :param noise: Noise variance.
    :param gamma_range: Number of kernels and hyperparameters.
    :param seed: Random seed.
    :return:
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate data
    X = np.linspace(-10, 10, n).reshape((n, 1))
    Xp = np.linspace(-10, 10, 500).reshape((500, 1))

    # Kernel sum
    Ksum = Kinterface(data=X, kernel=kernel_sum,
                      kernel_args={
                          "kernels": [exponential_kernel] * len(gamma_range),
                          "kernels_args": [{"gamma": g} for g in gamma_range]})

    # Sum of kernels
    Klist = [Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": g})
             for g in gamma_range]

    a = np.arange(n, dtype=int)
    if inducing_mode == "uniform":
        p = None
    elif inducing_mode == "biased":
        af = a.astype(float)
        p = (af ** 2 / (af ** 2).sum())
    else:
        raise ValueError(inducing_mode)

    inxs = np.random.choice(a, p=p, size=rank, replace=False)
    Kny = Ksum[:, inxs].dot(np.linalg.inv(Ksum[inxs, inxs])).dot(Ksum[inxs, :])
    f = mvn.rvs(mean=np.zeros((n,)), cov=Kny)
    y = mvn.rvs(mean=f, cov=noise * np.eye(n, n))

    return Ksum, Klist, inxs, X, Xp, y, f


def plot_signal(X, Xp, y, f, models=None, tit=""):
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
    :return:
    """

    # Plot signal
    plt.figure()
    x = X.ravel()
    xp = Xp.ravel()
    plt.plot(x, y, "k.")
    plt.plot(x, f, "r--")

    # Compute anchor ticks
    P = max([1] + map(lambda m: len(m.get("anchors", [])), models.values()))
    ymin = int(np.absolute(np.min(y)))
    Gxs = [np.linspace(-10, 10, 5 + 10 * g) for g in np.logspace(-1, 1, P)]
    Gys = range(-ymin - len(Gxs), -ymin)

    # Plot freqency scales
    for gi, (gx, gy) in enumerate(zip(Gxs, Gys)):
        plt.plot(gx, [gy] * len(gx), "|", color="gray")

    # Plot multiple signals and anchors
    if models is not None:
        for label, data in models.items():
            if label == "True": continue
            yp = data.get("yp", np.zeros((len(X), )))
            color = data.get("color", "blue")
            anchors = data.get("anchors", [[]])
            # rho, _ = pearsonr(yp, f)
            print(label, xp.shape, yp.shape)
            plt.plot(xp, yp, "-", color=color, label="%s" % label)
            for gi in range(P):
                if len(anchors) <= gi or not len(anchors[gi]): continue
                print("Number of pivots at gamma  %d: %d" % (gi, len(anchors[gi])))
                plt.plot(anchors[gi], [Gys[gi]] * len(anchors[gi]), "^", color=color, markersize=8, alpha=0.6)

    plt.title(tit)
    ylim = plt.gca().get_ylim()
    plt.legend()
    # plt.xlim((-x.min()-1, x.max()+1))
    plt.ylim((ylim[0]-1, ylim[1]))
    plt.show()


def test(Ksum, Klist, inxs, X, Xp, y, f, delta=10, lbd=0.1):
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
    :return:
    """
    P = len(Klist)                  # Number of kernels
    rank = len(inxs)      # Total number of inducing points over all lengthscales
    anchors = X[inxs,].ravel()

    # Fit MKL for kernel sum and
    mkl = Mklaren(rank=rank,
                  delta=delta, lbd=lbd)
    mkl.fit(Klist, y)
    y_Klist = mkl.predict([X] * len(Klist))
    yp_Klist = mkl.predict([Xp] * len(Klist))
    active_Klist = [mkl.data.get(gi, {}).get("act", []) for gi in range(P)]
    anchors_Klist = [X[ix] for ix in active_Klist]

    mkl.fit([Ksum], y)
    y_Ksum = mkl.predict([X])
    # yp_Ksum = mkl.predict([Xp])

    # Fit CSI
    csi = RidgeLowRank(rank=rank, lbd=lbd,
                       method="csi", method_init_args={"delta": delta},)
    csi.fit([Ksum], y)
    y_csi = csi.predict([X])
    yp_csi = csi.predict([Xp])
    active_csi = [csi.active_set_[gi] for gi in range(P)]
    anchors_csi = [X[ix] for ix in active_csi]

    # Fit ICD
    icd = RidgeLowRank(rank=rank, lbd=lbd,
                       method="icd")
    icd.fit([Ksum], y)
    y_icd = icd.predict([X])
    yp_icd = icd.predict([Xp])
    active_icd = [icd.active_set_[gi] for gi in range(P)]
    anchors_icd = [X[ix] for ix in active_icd]
    
    # Fit Nystrom
    nystrom = RidgeLowRank(rank=rank, lbd=lbd,
                       method="nystrom", method_init_args={"lbd": lbd, "verbose": False})
    nystrom.fit([Ksum], y)
    y_nystrom = nystrom.predict([X])
    yp_nystrom = nystrom.predict([Xp])
    active_nystrom = [nystrom.active_set_[gi] for gi in range(P)]
    anchors_nystrom = [X[ix] for ix in active_nystrom]

    # Correlation
    rho_Klist, _ = pearsonr(y_Klist, f)
    rho_Ksum, _ = pearsonr(y_Ksum, f)
    rho_csi, _ = pearsonr(y_csi, f)
    rho_icd, _ = pearsonr(y_icd, f)
    rho_nystrom, _ = pearsonr(y_nystrom, f)

    # Distance between anchors
    idp_dist_Klist = inducing_points_distance(anchors, anchors_Klist)
    idp_dist_CSI = inducing_points_distance(anchors, anchors_csi)
    idp_dist_icd = inducing_points_distance(anchors, anchors_icd)
    idp_dist_Nystrom = inducing_points_distance(anchors, anchors_nystrom)

    # Plot a summary figure
    return {"True": {"anchors": anchors,
                     "color": "black"},
            "Mklaren": {
                 "rho": rho_Klist,
                 "active": active_Klist,
                 "anchors": anchors_Klist,
                 "idp": idp_dist_Klist,
                 "yp": yp_Klist,
                 "color": "green"},
            "CSI": {
                "rho": rho_csi,
                "active": active_csi,
                "anchors": anchors_csi,
                "idp": idp_dist_CSI,
                "yp": yp_csi,
                "color": "blue"},
            "ICD": {
                "rho": rho_icd,
                "active": active_icd,
                "anchors": anchors_icd,
                "idp": idp_dist_icd,
                "yp": yp_icd,
                "color": "cyan"},
            "Nystrom": {
                "rho": rho_nystrom,
                "active": active_nystrom,
                "anchors": anchors_nystrom,
                "idp": idp_dist_Nystrom,
                "yp": yp_nystrom,
                "color": "pink"}
            }

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

def main():
    # Experiment paramaters
    n_range = (100, )
    rank_range = (3, 5, 10)
    lbd_range = (0, )
    gamma_range = [0.1, 0.3, 1, 3]
    repeats = 500
    pc = 0.1 # pseudocount; prevents inf in KL-divergence.
    noise_models = ("fixed", "increasing")
    sampling_models = ("uniform", "biased")
    methods = ("Mklaren", "CSI", "ICD", "Nystrom")

    # Create output directory
    d = datetime.datetime.now()
    dname = os.path.join("output", "snr", "%d-%d-%d" % (d.year, d.month, d.day))
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
              "anchors.dist", "anchors.dist.sd", "total.variation", "kl.divergence"]
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
        if noise_model == "fixed":
            noise = 1
        else:
            noise = np.logspace(-2, 2, n)

        avg_actives = dict()
        avg_anchors = dict()
        avg_dists = dict()

        r = None
        for seed in range(repeats):
            # Generate data
            Ksum, Klist, inxs, X, Xp, y, f = generate_data(n=n,
                                                           rank=rank,
                                                           inducing_mode=inducing_mode,
                                                           noise=noise,
                                                           gamma_range=[gamma],
                                                           seed=seed)
            # Evaluate methods
            try:
                r = test(Ksum, Klist, inxs, X, Xp, y, f)
            except Exception as e:
                print("Exception, continuing: %s" % str(e))
                continue

            # Fill in anchors and active points
            avg_actives["True"] = avg_actives.get("True", []) + list(inxs)
            avg_anchors["True"] = avg_anchors.get("True", []) + list(r["True"]["anchors"])
            for m in methods:
                avg_actives[m] = avg_actives.get(m, []) + list(r[m]["active"][0])
                avg_anchors[m] = avg_anchors.get(m, []) + list(r[m]["anchors"][0])
                avg_dists[m] = avg_dists.get(m, []) + [r[m]["idp"]]

        # Compare distributions
        rows = []
        probs, bins  = np.histogram(avg_actives["True"], normed=False)
        probs = 1.0 * (probs + pc) / (probs + pc).sum()
        for m in methods:
            query, _ = np.histogram(avg_actives[m], normed=False, bins=bins)
            query = 1.0 * (query + pc ) / (query + pc).sum()
            kl = entropy(probs, query)
            tv = hist_total_variation(probs, query)
            idp_mean = np.mean(avg_dists[m])
            idp_std = np.std(avg_dists[m])
            row  = {"method": m, "noise.model": noise_model, "sampling.model": inducing_mode,
                    "n": n, "rank":rank, "lbd": lbd, "gamma": gamma,
                    "anchors.dist": idp_mean, "anchors.dist.sd": idp_std,
                    "total.variation": tv, "kl.divergence": kl}
            rows.append(row)

            # Extended row for details
            row_extd = row.copy()
            row_extd["avg.dists"] = avg_dists[m]
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
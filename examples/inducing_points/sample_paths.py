hlp = """
    Draw sample paths from degenerate Gaussian Processes, to compare the selection of inducing points
    by different methods.
"""

from examples.inducing_points.inducing_points import *

# Hypereparamters
n = 100                                 # Number of data points
rank = 3                                # Approximation rank
inducing_model = "biased"               # Fixed sampling regime
noise = 0.03                            # Fixed noise
seed_range = xrange(41, 41 + 30)        # Random seeds
seed_plot_range = [41, 44, 45]          # Selected examples for plotting


def process(outdir):
    """
    Run experiments with epcified parameters.
    :param outdir: Output directory.
    :return:
    """

    # Output directory
    if not os.path.exists(outdir): os.makedirs(outdir)

    # Output
    header = ["method", "seed", "rho", "evar"]
    fp = open(os.path.join(outdir, "samples_paths.csv"), "w")
    writer = csv.DictWriter(fp, fieldnames=header)
    writer.writeheader()

    results = dict()
    data = dict()
    for seed in seed_range:
        Ksum, Klist, inxs, X, Xp, y, f = generate_data(n=n,
                                                       rank=rank,
                                                       inducing_mode=inducing_model,
                                                       noise=noise,
                                                       gamma_range=[1.0],
                                                       seed=seed,
                                                       input_dim=1)
        data[seed] = (X, y, f)

        # Compute models
        r = test(Ksum, Klist, inxs, X, Xp, y, f, methods=["Mklaren", "CSI"])
        for method in ["CSI", "Mklaren"]:
            row = {"method": method, "seed": seed,
                   "rho": "%.3f" % r[method]["rho"], "evar": "%.3f" % r[method]["evar"]}
            writer.writerow(row)

        # Generate Mklaren solution path
        sol_path_mkl = r["Mklaren"]["sol_path"]
        sol_anchors_mkl = r["Mklaren"]["active"]

        # Generate CSI solution path
        sol_path_csi = []
        sol_anchors_csi = None
        for rk in range(1, rank+1):
            models = test(Ksum, Klist, inxs[:rk], X, Xp, y, f, methods=["CSI"], kappa=0.99999)
            sol_path_csi.append(models["CSI"]["yp"].ravel())
            sol_anchors_csi = r["CSI"]["active"]

        # Store results
        results[seed] = {"CSI": (sol_path_csi, sol_anchors_csi),
                         "Mklaren": (sol_path_mkl, sol_anchors_mkl)}

    # Plot a composite figure
    cols = ["Mklaren", "CSI"]
    fig, axes = plt.subplots(figsize=(4.72, 5.51),
                             ncols=2, nrows=len(seed_plot_range),
                             sharex=False, sharey=False)

    # Plot for selected cases
    for i, seed in enumerate(seed_plot_range):
        dd = results[seed]
        X, y, f = data[seed]
        for method, (sol_path, sol_anchors) in dd.iteritems():
            j = cols.index(method)
            ax = axes[i][j]
            if i == 0: ax.set_title(method)
            if i == len(seed_plot_range) - 1: ax.set_xlabel("Input space (x)")
            if j == 0: ax.set_ylabel("Output space (y)")
            ax.plot(X.ravel(), y, ".", color="gray")
            ax.plot(X.ravel(), f, "--", color="gray")

            # Inducing inputs
            locs = X.ravel()[sol_anchors[0]]
            labs = map(lambda l: "%s." % l, np.arange(len(locs), dtype=int) + 1)
            ax.set_xticks(locs)
            ax.set_xticklabels(labs)
            ymin = ax.get_ylim()[0]
            ax.plot(locs, [ymin]*len(locs), "^", color=meth2color[method], markersize=10)

            # Solution paths
            for si, sol in enumerate(sol_path):
                ax.plot(X.ravel(), sol.ravel(), label=str(si), linewidth=1.5*(si+1),
                         color=meth2color[method], alpha=0.35)

    # Close output
    fig.tight_layout()
    for ext in ("pdf", "eps"):
        fname = os.path.join(outdir, "sample_paths.%s" % ext)
        plt.savefig(fname, bbox_inches="tight")
        print("Written %s" % fname)
    plt.close()
    fp.close()


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description=hlp)
    parser.add_argument("output",  help="Output directory.")
    args = parser.parse_args()

    # Output directory
    odir = args.output
    process(odir)
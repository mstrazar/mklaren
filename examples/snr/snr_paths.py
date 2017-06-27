from examples.snr.snr import *

# Fixed output
out_dir = "/Users/martin/Dev/mklaren/examples/output/snr/solution_paths"
assert os.path.exists(out_dir)

# Output
header = ["method", "seed", "rho", "evar"]
fp = open(os.path.join(out_dir, "results.csv"), "w")
writer = csv.DictWriter(fp, fieldnames=header)
writer.writeheader()

n = 100
rank = 3
inducing_model = "biased"
noise = 0.03
for seed in xrange(42, 48):

    Ksum, Klist, inxs, X, Xp, y, f = generate_data(n=n,
                                                   rank=rank,
                                                   inducing_mode=inducing_model,
                                                   noise=noise,
                                                   gamma_range=[1.0],
                                                   seed=seed,
                                                   input_dim=1)

    # Compute models
    r = test(Ksum, Klist, inxs, X, Xp, y, f, methods=["Mklaren", "CSI"])
    # plot_signal(X, Xp, y, f, models=r, tit="")
    for method in ["CSI", "Mklaren"]:
        row = {"method": method, "seed": seed,
               "rho": "%.3f" % r[method]["rho"], "evar": "%.3f" % r[method]["evar"]}
        writer.writerow(row)

    # Generate Mklaren solution path
    sol_path_mkl = r["Mklaren"]["sol_path"]
    sol_anchors_mkl = r["Mklaren"]["active"]

    # Generate CSI solution path
    sol_path_csi = []
    for rk in range(1, rank+1):
        models = test(Ksum, Klist, inxs[:rk], X, Xp, y, f, methods=["CSI"], kappa=0.99999)
        sol_path_csi.append(models["CSI"]["yp"].ravel())
        sol_anchors_csi = r["CSI"]["active"]

    # Plot Mklaren solution paths
    for sol_name, sol_path, sol_anchors in zip(["Mklaren", "CSI"],
                                                [sol_path_mkl, sol_path_csi],
                                                [sol_anchors_mkl, sol_anchors_csi]):
        fname = os.path.join(out_dir, "sol_path_%s-seed-%d.pdf" % (sol_name, seed))
        plt.figure(figsize=(3.38, 2.0))
        plt.title(sol_name)
        plt.plot(X.ravel(), y, ".", color="gray")
        plt.plot(X.ravel(), f, "--", color="gray")
        for si, sol in enumerate(sol_path):
            ai = sol_anchors[0][si]
            plt.plot(X.ravel(), sol.ravel(), label=str(si), linewidth=1.5*(si+1),
                     color=meth2color[sol_name], alpha=0.5)
            plt.text(X.ravel()[ai], sol.ravel()[ai] + 0.05, str(si))
        plt.xlabel("Input space (x)")
        plt.ylabel("Output space (y)")
        # plt.legend(loc=2)
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print("Written %s" % fname)


# Close output
fp.close()

from examples.snr import *



def plot_results(results, f_out=None, tit=""):

    plt.figure()

    for mi, (label, data) in enumerate(sorted(results.items())):
        counts, bins = np.histogram(data, normed=False)
        probs = 1.0 * counts / counts.sum()
        cs = np.cumsum(probs)
        xs = bin_centers(bins)

        fm = "--" if label == "True" else "-"
        color = "black" if label == "True" else "green"
        linewidth = 1 if label == "True" else 1 + 1.2 * mi
        plt.plot([-10] + list(xs), [0] + list(cs), fm,
                 label=label, linewidth=linewidth, color=color)

    plt.ylabel("Cumulative probability")
    plt.xlabel("Inducing point (pivot) location")
    plt.legend(loc=2)
    plt.title(tit)
    if f_out is None:
        plt.show()
    else:
        plt.savefig(f_out, bbox_inches="tight")
        plt.close()
        print("Written %s" % f_out)


def main():
    out_dir = "output/snr/regularization/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    seed_range = range(100)
    n = 100
    lambda_range = [0, 0.1, 0.3, 1]
    results = dict()

    for noise_model, inducing_model in it.product(("fixed", "increasing"),
                                                  ("biased", "uniform")):
        for lbd, seed in it.product(lambda_range, seed_range):
            noise = generate_noise(n, noise_model, 1)
            Ksum, Klist, inxs, X, Xp, y, f = generate_data(n=n,
                                                       rank=5,
                                                       inducing_mode=inducing_model,
                                                       noise=noise,
                                                       gamma_range=[1.0],
                                                       seed=seed,
                                                       input_dim=1)
            r = test(Ksum, Klist, inxs, X, Xp, y, f, lbd=lbd)

            tru = r["True"]["anchors"]
            results["True"] = results.get("True", []) + list(tru.ravel())

            ky = ("Mklaren $\lambda=%.1f$ " % lbd)
            samp = r["Mklaren"]["anchors"]
            results[ky] = results.get(ky, []) + list(samp[0].ravel())

        fname = os.path.join(out_dir, "lambda_effect_%s_%s.pdf" % (noise_model, inducing_model))
        tit = "Noise: %s, sampling: %s" % (noise_model, inducing_model)
        plot_results(results, f_out=fname, tit=tit)





if __name__ == "__main__":
    main()
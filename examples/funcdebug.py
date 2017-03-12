from examples.functions import *
import matplotlib.pyplot as plt

repeats = 5
ns = [100, 300, 1000]
P = 5
ranks = [5, 10, 20, 50, 80, 90]
lbds = [0, 1, 10, 100]

for repl, n in it.product(range(repeats), ns):
    np.random.seed(repl)

    # Output
    outdir = os.path.join("output", "funcdebug", "%d_%d" % (n, repl))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data = generate_data(P=P, n=n, row_normalize=True)
    mf = data["mf"]
    Ks, names = data_kernels(data["X"], mf, row_normalize=True, noise=0.01)
    true_w = mf.to_dict()
    true_result = true_w.copy()
    print(repl, n)
    print(mf)

    # Store data on current combination
    fp = open(os.path.join(outdir, "data.txt"), "w")
    fp.write(str(mf))
    fp.close()

    # Select one exponential kernel and a full set
    Ks, names = data_kernels(data["X"], mf, row_normalize=True)
    Ks1, names1 = [Ks[0]], [names[0]]

    # MSEs
    m1 = np.zeros((len(lbds), len(ranks)))
    m2 = np.zeros((len(lbds), len(ranks)))

    # Number of different kernels within the model (only applicable to full model)
    nk1 = np.zeros((len(lbds), len(ranks)))
    prec1 = np.zeros((len(lbds), len(ranks)))
    rho1 = np.zeros((len(lbds), len(ranks)))

    for li, lbd in enumerate(lbds):
        for ri, r in enumerate(ranks):
            # Full model
            mklaren1 = mkl.mkl.mklaren.Mklaren(delta=10, rank=r, lbd=lbd)
            mklaren1.fit(Ks, data["y"])
            m1[li, ri] = mean_squared_error(data["y"], mklaren1.y_pred)
            nk1[li, ri] = len(set(mklaren1.G_mask.ravel()))

            w1 = weight_result(names, mklaren1.mu)
            p, _ = weight_PR(d_true=true_w, d_pred=w1)
            prec1[li, ri] = p

            rh, _ = weight_correlation(true_w, w1, typ="kendall")
            rho1[li, ri] = rh

            # Reduced model
            mklaren2 = mkl.mkl.mklaren.Mklaren(delta=10, rank=r, lbd=lbd)
            mklaren2.fit(Ks1, data["y"])
            m2[li, ri] = mean_squared_error(data["y"], mklaren2.y_pred)


    # Dependence between rank, regularization and model fit.
    # In principle, the exponential is a full rank kernel matrix and should be able to fit any data.
    # The full model is at a disadvantage here, because ot has more to choose from (and the lookahead
    # heuristic is not perfect). However, by adding regularization, the full model produces better
    # model fits, and the gap grows with larger feature space size.
    plt.figure()
    for li, lbd in enumerate(lbds):
        plt.plot(m1[li, :], label="full ($\lambda$=%d)" % lbd, linewidth=(1+li), color="blue")
        plt.plot(m2[li, :], label="reduced ($\lambda$=%d)" % lbd, linewidth=(1+li), color="green")
    plt.xticks(range(len(ranks)))
    plt.gca().set_xticklabels(ranks)
    plt.xlabel("Rank")
    plt.ylabel("MSE")
    plt.legend(loc=3)
    plt.savefig(os.path.join(outdir, "mse.pdf"))
    plt.close()

    # Number of different kernels
    plt.figure()
    for li, lbd in enumerate(lbds):
        plt.plot(nk1[li, :], "o-", label="full ($\lambda$=%d)" % lbd, linewidth=(1+li))
    plt.xticks(range(len(ranks)))
    plt.gca().set_xticklabels(ranks)
    plt.xlabel("Rank")
    plt.ylabel("Num. kernels")
    plt.legend(loc=4)
    plt.savefig(os.path.join(outdir, "nk.pdf"))
    plt.close()

    # Number of different kernels
    plt.figure()
    for li, lbd in enumerate(lbds):
        plt.plot(prec1[li, :], "o-", label="full ($\lambda$=%d)" % lbd, linewidth=(1+li))
    plt.xticks(range(len(ranks)))
    plt.gca().set_xticklabels(ranks)
    plt.xlabel("Rank")
    plt.ylabel("Precision")
    plt.legend(loc=1)
    plt.savefig(os.path.join(outdir, "prec.pdf"))
    plt.close()

    # Number of different kernels
    plt.figure()
    for li, lbd in enumerate(lbds):
        plt.plot(rho1[li, :], "o-", label="full ($\lambda$=%d)" % lbd, linewidth=(1+li))
    plt.xticks(range(len(ranks)))
    plt.gca().set_xticklabels(ranks)
    plt.xlabel("Rank")
    plt.ylabel("Kendall rho")
    plt.legend(loc=2)
    plt.savefig(os.path.join(outdir, "rho.pdf"))
    plt.close()


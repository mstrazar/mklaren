import scipy.stats as st
import matplotlib.pyplot as plt
import pickle

from collections import Counter
from mklaren.kernel.string_kernel import *
from mklaren.kernel.string_util import *
from mklaren.kernel.kernel import kernel_sum
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank



def generic_function_plot(f_out, Ks, X, models=(),
                          sample_size=1000, seed=None,
                          title="", xnames=(), xlabel="", truePar=None):
    """
    Plot a general function in any input space. Depending on (typically many) kernel hyperparameters,
    see how distances in corresponding RKHS, match the distances in the predicted output.
    This way, a function in arbitrary input space can approximately be assessed in terms
    of kernel hyperparameters (lengthscales, k-mer lengths etc).

    :param f_out: Output figure file.
    :param Ks: List of kernel interfaces.
    :param X: Points in the input space.
    :param models: Dictionary containing model predictions.
    :param title: Optional
    :param seed: Random seed.
    :param xnames: Names for parameter range.
    :param xlabel: X label.
    :param truePar: True INDEX of parameter values.
    :param sample_size: Number of sample pairs.
    :return:
    """
    if seed:
        np.random.seed(seed)

    # Range of parameter names
    par_range = range(len(Ks))
    assert len(xnames) == 0 or len(xnames) == len(par_range)

    # Random sample of pairs in test set
    n = len(X)
    samp1 = np.random.choice(range(n), size=sample_size, replace=True)
    samp2 = np.random.choice(range(n), size=sample_size, replace=True)

    # Plot some sort of correlation between distance in the feature and output space
    # Select random pairs of points from the test set
    corrs = dict()

    for ki in par_range:
        # Distances in feature space on
        kern =  Ks[ki].kernel
        kargs = Ks[ki].kernel_args
        Di = np.array([np.sqrt(-2 * kern(X[i], X[j], **kargs) \
                                  + kern(X[i], X[i], **kargs) \
                                  + kern(X[j], X[j], **kargs)) for i, j in zip(samp1, samp2)])

        # Distances in output space on sample
        for label, data in models.items():
            y = data["y"]
            yd = np.absolute(np.array([y[i] - y[j] for i, j in zip(samp1, samp2)]))
            pc = st.pearsonr(Di.ravel(), yd.ravel())
            corrs[label] = corrs.get(label, []) + [pc[0]]

    # Plot a summary figure
    plt.figure()
    plt.title(title)
    for label, pc_vec in corrs.items():
        kwargs = {"label": label}
        color = models[label].get("color", None)
        fmt = models[label].get("fmt", None)
        if color: kwargs["color"] = color
        if fmt: kwargs["linestyle"] = fmt
        plt.plot(par_range, pc_vec, linewidth=2, **kwargs)

    # X
    plt.ylabel("Pearson correlation $d(i, j)$, $|y_i-y_j|$")
    ylim = plt.gca().get_ylim()
    if truePar is not None:
        plt.plot((truePar, truePar), ylim, "-", color="black", label="True hyperpar.")
    plt.ylim(ylim)

    # Y
    plt.xlabel(xlabel)
    plt.xticks(par_range)
    plt.xlim(par_range[0]-0.5, par_range[-1]+0.5)
    if len(xnames): plt.gca().set_xticklabels(map(str, xnames))
    plt.grid("on")
    plt.legend(loc="best")
    plt.savefig(f_out)
    plt.close()

    obj = {"f_out": f_out,
           "Ks": Ks, "X": X, "models": models,
           "sample_size": sample_size, "seed": seed,
           "title": title, "xnames": xnames, "xlabel": xlabel, "truePar": truePar}
    pickle.dump(obj, open(f_out + ".pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)
    print "Written %s" % f_out
    return


def process():

    # Experimental parameters
    rank = 3
    delta = 10
    lbd = 0
    L = 30
    N = 50
    trueK = 4
    max_K = 10
    K_range = range(1, max_K+1)
    normalize = False

    # Random subset of N sequences of length L
    X, _ = generate_data(N=N, L=L, p=0.0, motif="TGTG", mean=0, var=3)
    X = np.array(X)

    # Generate a sparse signal based on 4-mer composion (maximum lengthscale)
    K = Kinterface(data=X, kernel=string_kernel, kernel_args={"mode": SPECTRUM, "K": trueK},
                   row_normalize=normalize)
    y = st.multivariate_normal.rvs(mean=np.zeros((N,)), cov=K[:, :]).reshape((N, 1))
    yr = y.ravel()

    # Proposal kernels
    args = [{"mode": SPECTRUM, "K": k} for k in K_range]
    Ksum = Kinterface(data=X, kernel=kernel_sum,
                          row_normalize=normalize,
                          kernel_args={"kernels": [string_kernel] * len(args),
                                       "kernels_args": args})
    Ks = [Kinterface(data=X, kernel=string_kernel, kernel_args=a, row_normalize=normalize) for a in args]

    # Mklaren
    mklaren = Mklaren(rank=rank, delta=delta, lbd=lbd)
    mklaren.fit(Ks, y)
    yp_mkl = mklaren.predict([X]*len(args)).ravel()
    mklaren_kernels = [(args[int(ky)]["K"], val) for ky, val in sorted(Counter(mklaren.G_mask).items())]
    for lg, num in sorted(mklaren_kernels, key=lambda t:t[1], reverse=True):
        print "K: %d (%d)" % (lg, num)


    # CSI
    csi = RidgeLowRank(rank=rank, method="csi",
                       method_init_args={"delta": delta}, lbd=lbd)
    csi.fit([Ksum], y)
    yp_csi = csi.predict([X]).ravel()


    # Print data along with predictions
    for xi, yi, ym, yc in sorted(zip(X, y, yp_mkl, yp_csi), key=lambda t: t[1]):
        print "%s\t%.3f\t%.3f\t%.3f" % (xi, yi, ym, yc)

    # Spearman correlation fo the fit
    print "\nMklaren fit: %.3f (%.5f)" % st.spearmanr(y, yp_mkl)
    print "CSI fit: %.3f (%.5f)" % st.spearmanr(y, yp_csi)

    # Represent fit on figure
    plt.figure()
    plt.plot(yr, yp_mkl, ".", color="green", label="Mklaren")
    plt.plot(yr, yp_csi, ".", color="blue", label="CSI")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.legend()


    # Spearman correlation fo the fit
    print "\nMklaren residual corr.: %.3f (%.5f)" % st.spearmanr(yr, yr-yp_mkl,)
    print "CSI residual corr.: %.3f (%.5f)" % st.spearmanr(yr, yr-yp_csi)
    print

    # Residual graph
    plt.figure()
    plt.plot(yr, yr-yp_mkl, ".", color="green", label="Mklaren")
    plt.plot(yr, yr-yp_csi, ".", color="blue", label="CSI")
    plt.xlabel("True")
    plt.ylabel("Residual")
    plt.legend()


    # fname = "/Users/martin/Dev/mklaren/examples/output/string/lengthscales_%d_1.pdf" % trueK
    fname = "/Users/martin/Dev/mklaren/examples/output/string/test.pdf"
    generic_function_plot(f_out=fname, Ks=Ks, X=X,
                          models={"True": {"y": yr, "color": "black", "fmt": "--",},
                                  "Mklaren": {"y": yp_mkl, "color": "green", "fmt": "-",},
                                  "CSI": {"y": yp_csi, "color": "red", "fmt": "-"}},
                          xlabel="K-mer length",
                          xnames=K_range,
                          truePar=K_range.index(trueK))


if __name__ == "__main__":
    process()
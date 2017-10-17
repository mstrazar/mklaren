# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import scipy.stats as st
import matplotlib.pyplot as plt
import pickle
from mklaren.kernel.string_util import *


def generic_function_plot(f_out, Ks, X, models=(),
                          sample_size=5000, seed=None,
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
    plt.figure(figsize=(3.2, 2.75))
    plt.title(title)
    for label, pc_vec in corrs.items():
        kwargs = {"label": label.replace("Nystrom", "Nyström")}
        color = models[label].get("color", None)
        fmt = models[label].get("fmt", None)
        if color: kwargs["color"] = color
        if fmt: kwargs["linestyle"] = fmt
        plt.plot(par_range, pc_vec, linewidth=2, **kwargs)

    # X
    plt.ylabel("Pearson correlation $d(i, j)$, $|yp_i-yp_j|$")
    ylim = plt.gca().get_ylim()
    if truePar is not None:
        plt.plot((truePar, truePar), ylim, "-", color="black")
    plt.ylim(ylim)

    # Y
    plt.xlabel(xlabel)
    plt.xticks(par_range)
    plt.xlim(par_range[0]-0.5, par_range[-1]+0.5)
    if len(xnames): plt.gca().set_xticklabels(map(str, xnames))
    plt.grid("on")
    plt.legend(ncol=3, loc=(0, 1.1), frameon=False)
    plt.savefig(f_out, bbox_inches="tight")
    plt.close()

    obj = {"f_out": f_out,
           "Ks": Ks, "X": X, "models": models,
           "sample_size": sample_size, "seed": seed,
           "title": title, "xnames": xnames, "xlabel": xlabel, "truePar": truePar}
    pickle.dump(obj, open(f_out + ".pkl", "w"), protocol=pickle.HIGHEST_PROTOCOL)
    print "Written %s" % f_out

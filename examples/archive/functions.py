# Temporary fix for octave executable
import os
octv = "/usr/local/octave/3.8.0/bin/octave-3.8.0"
if os.path.exists(octv):
    os.environ["OCTAVE_EXECUTABLE"] = octv

from mklaren.kernel.kernel import exponential_kernel, linear_kernel, poly_kernel, center_kernel, \
    matern_kernel, periodic_kernel, kernel_row_normalize
from scipy.stats import chi2, spearmanr, kendalltau, pearsonr
from sklearn.metrics import precision_score, recall_score, mean_squared_error
from random import sample
import numpy as np
import mklaren as mkl
import itertools as it
import csv
import datetime


class MultiKernelFunction:

    # Library of function families
    library = {
        "exp": (exponential_kernel, ("gamma", )),
        "lin": (linear_kernel, ("b",)),
        "pol": (poly_kernel, ("degree", )),
        "mat": (matern_kernel, ("nu", "l")),
        # "per": (periodic_kernel, ("per", "l"))
    }

    # Hyper parameter ranges
    values = {
        "gamma": np.logspace(-3, 1, 5),
        "degree": [2, 3, 4, 5, 6],
        # "per": [1, 10, 30],
        "l": np.logspace(0, 2, 3),
        "b": [0, 1],
        "nu": [1.5, 2.0, 2.5],
    }

    @staticmethod
    def name(f, args):
        """
        Name a kernel.
        :param f:
        :param pars:
        :return:
        """
        a = ", ".join("%s=%.2f" % (k, v) for k, v in args.items())
        return "%s (x, x, %s)" % (f, a)


    def kernel_generator(self, interface=False):
        """
        Generate the set of all possible kernel given by parameters.
        :return:
            A listing of all kernel objects.
        """
        listing = list()
        for name, (f, pars) in sorted(self.library.items()):
            par_value_lists = [list(it.product((p,), self.values[p])) for p in pars]
            instances = it.product(*par_value_lists)

            if interface:
                lst = [("k.%s.%d" % (name, i), f, dict(kw)) for i, kw in enumerate(instances)]
            else:
                lst = [("k.%s.%d" % (name, i), f, dict(kw)) for i, kw in enumerate(instances)]
            listing.extend(lst)
        return listing


    def __init__(self, p, center=False, row_normalize=False):
        """
        Initialize a random combination of kernels.
        :param p: Number of kernels.
        """
        assert not (center and row_normalize)
        self.p = p
        self.center = center
        self.row_normalize = row_normalize

        # Sample a set of non-repeating kernel definitions and unzip
        kgen = self.kernel_generator()
        result = sample(kgen, p)
        self.signs, self.funcs, self.args = zip(*result)

        # Sample random kernel weights
        self.weights = np.array([chi2.rvs(df=1) for pi in range(p)])


    def kernel_matrix(self, X):
        """
        Compute a combined kernel matrix on the fly.
        :param X:
            Data matrix.
        :return:
        """
        n = np.array(X).shape[0]
        K = np.zeros((n, n))

        for w, f, args in zip(self.weights, self.funcs, self.args):
            L = f(X, X, **args)
            if self.center:
                L = center_kernel(L)
            elif self.row_normalize:
                L = kernel_row_normalize(L)
            K = K + w * L
        return K


    def __call__(self, X, alpha):
        """
        Return the signal encoded by the kernel function and dual coefficients alpha.
        :param X:
            A data matrix.
        :param alpha:
            Dual coefficients.
        :return:
        """
        n = np.array(X).shape[0]
        assert len(alpha) == n
        alpha = alpha.reshape((n, 1))
        K = self.kernel_matrix(X)
        return K.dot(alpha)


    def __str__(self):
        """
        Print the rule encoded by this function
        :return:
            A text representation of a random kernel function.
        """
        tups = sorted(zip(self.weights, self.signs, self.args), reverse=True)
        mp = map(lambda t: "%0.2e %s(x, x, %s)" % t, tups)
        txt = " + \n\t ".join(mp)
        return "y = \n\t %s" % txt


    def to_dict(self):
        """
        Represent combination of kernels as a mapping of names to weights.
        :return:
            Numpy array indexed by names.
        """
        am = lambda pi: (self.signs[pi], self.weights[pi])
        mp = map(am, range(self.p))
        return dict(mp)



def generate_data(P=3, p=10, n=300, row_normalize=False):
    """
    Generate a random dataset.
    :param P:
        Number of kernels.
    :param p:
        Number of dimensions.
    :param n:
        Number of data examples.
    :return:
    """
    # Data is generated randomly
    X = np.random.randn(n, p)

    # Dual coefficients are non-negative
    alpha = np.array([np.random.randn() for i in range(n)])

    mf = MultiKernelFunction(P, row_normalize=row_normalize)
    y = mf(X, alpha)
    y = y - y.mean()

    return {"X": X,
            "y": y,
            "alpha": alpha,
            "mf": mf}


def data_kernels(X, mf, center=False, row_normalize=False, noise=0):
    """
    Map data to all possible kernels given by MultiKernelFunction object.
    :param X:
        Data matrix.
    :param mf:
        MultiKernelFunction object definition.
    :param center
        Center the kernel matrix.
    :param row_normalize
        Apply feature space vector normalization.
    :param noise
        Add noise to ensure full rank kernels.
    :return:
        List of kernel matrices.
    """
    assert not(center and row_normalize)
    Ks = list()
    names = list()
    for name, f, args in mf.kernel_generator():
        L = f(X, X, **args) + noise * np.eye(X.shape[0])
        if center:
            L = center_kernel(L)
        elif row_normalize:
            L = kernel_row_normalize(L)
        Ks.append(L)
        names.append(name)
    return Ks, names


def weight_result(names, weights):
    """Create a dictionary of retrieved kernels and weights"""
    return dict([(n, w) for n, w in zip(names, weights) if w != 0])


def weight_correlation(d1, d2, typ="spearman"):
    """Measure agreement between two sets of weights in terms of Spearman rho."""
    names = sorted(set(d1.keys()) | set(d2.keys()))
    arr1 = np.zeros((len(names),))
    arr2 = np.zeros((len(names),))
    for i, n in enumerate(names):
        arr1[i] += d1.get(n, 0)
        arr2[i] += d2.get(n, 0)
    if typ == "spearman":
        return spearmanr(arr1, arr2)
    elif typ == "kendall":
        return kendalltau(arr1, arr2)
    elif typ == "pearson":
        return pearsonr(arr1, arr2)
    else:
        raise ValueError(typ)



def weight_PR(d_true, d_pred):
    """Measure agreement between two sets of weights in terms of Spearman rho."""
    names = sorted(set(d_true.keys()) | set(d_pred.keys()))
    arr1 = np.zeros((len(names),))
    arr2 = np.zeros((len(names),))
    for i, n in enumerate(names):
        arr1[i] += int(d_true.get(n, 0) > 0)
        arr2[i] += int(d_pred.get(n, 0) > 0)
    p = precision_score(arr1, arr2)
    r = recall_score(arr1, arr2)
    return p, r


def process(repeats, fname):
    """
    :param repeats:
        Number of repetitions.
    :return:
        Simulation results for different methods.


    Ways to compare results in this case:
        - Reconstruct a kernel matrix via regression using all the methods. This is reminiscent of the tru situation,
        where we don't know what sort of relations were used to generate the data.
        - Count the intersection of correctly retrieved kernel families.
        - Measure Spearman rank correlation of obtained versus true weights for appropriate kernels.


    """
    # Open an output file
    header = ["exp.id", "repl", "P", "n", "rank", "lbd", "method", "prec", "recall", "MSE",
              "spearman_rho", "spearman_pvalue",
              "pearson_rho", "pearson_pvalue",
              "kendall_rho", "kendall_pvalue",]
    names = [nm for nm, _, _ in MultiKernelFunction(1).kernel_generator()]
    header.extend(names)
    writer = csv.DictWriter(open(fname, "w", buffering=0),
                            fieldnames=header, quoting=csv.QUOTE_ALL)
    writer.writeheader()

    # Varying parameters
    range_P = [3, 5]
    range_rank = [3, 5, 10]
    range_n = [100, 300, 1000]
    range_lbd = [0] + np.logspace(-1, 1, 3)

    count = 0
    for repl in range(repeats):
        for P, n, rank, lbd in it.product(range_P, range_n, range_rank, range_lbd):
            row = {"exp.id": count, "repl": repl, "P": P, "n": n, "rank": rank}
            count += 1

            # Generate test data
            data = generate_data(P=P, n=n, row_normalize=True)
            mf = data["mf"]
            Ks, names = data_kernels(data["X"], mf, row_normalize=True, noise=0.01)
            true_w = mf.to_dict()
            true_result = true_w.copy()
            true_result.update(row)

            print("Original kernel")
            print(mf)

            rows = []
            # for method in ["Mklaren", "ICD", "CSI"]:
            for method in ["Mklaren", "ICD",]:
                if rows is None: break
                print("Fitting %s ... (%s)" % (method, row))

                try:
                    if method == "Mklaren":
                        effective_rank = P * rank
                        mklaren = mkl.mkl.mklaren.Mklaren(delta=10, rank=effective_rank, lbd=lbd)
                        mklaren.fit(Ks, data["y"])
                        results_w = weight_result(names, mklaren.mu)
                        y_pred = mklaren.y_pred
                    elif method == "ICD":
                        effective_rank = rank
                        icd = mkl.regression.ridge.RidgeLowRank(rank=effective_rank, lbd=lbd, method="icd")
                        icd.fit(Ks, data["y"])
                        results_w = weight_result(names, icd.mu)
                        y_pred = icd.y_pred
                    elif method == "CSI":
                        effective_rank = rank
                        csi = mkl.regression.ridge.RidgeLowRank(rank=effective_rank, lbd=lbd,
                                                                method="csi",
                                                                method_init_args={"delta": 10})
                        csi.fit(Ks, data["y"])
                        results_w = weight_result(names, csi.mu)
                        y_pred = csi.y_pred

                except Exception as e:
                    print "%s error: %s" % (method, e)
                    rows = None
                    continue

                pr = weight_PR(true_w, results_w)
                mse = mean_squared_error(y_true=data["y"], y_pred=y_pred)
                results_row = {"method": method, "MSE": mse, "lbd": lbd,
                                  "prec": pr[0], "recall": pr[1]}
                for t in ["spearman", "kendall", "pearson"]:
                    wc = weight_correlation(results_w, true_w, typ=t)
                    results_row["%s_rho" % t] = wc[0]
                    results_row["%s_pvalue" % t] = wc[1]
                results_row.update(row)
                results_row.update(results_w)
                if rows is not None: rows.append(results_row)

            # Append if all methods go trough
            if rows is not None:
                writer.writerow(true_result)
                writer.writerows(rows)

if __name__  == "__main__":
    d = datetime.datetime.now()
    dname = os.path.join("output", "functions", "%d-%d-%d" % (d.year, d.month, d.day))
    if not os.path.exists(dname):
        os.makedirs(dname)
    fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
    process(100, fname)


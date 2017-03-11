# Temporary fix for octave executable
import os
octv = "/usr/local/octave/3.8.0/bin/octave-3.8.0"
if os.path.exists(octv):
    os.environ["OCTAVE_EXECUTABLE"] = octv

from mklaren.kernel.kernel import exponential_kernel, linear_kernel, poly_kernel, center_kernel, \
    matern_kernel, periodic_kernel, kernel_row_normalize
from scipy.stats import chi2, spearmanr
from sklearn.metrics import precision_score, recall_score
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
        "gamma": np.logspace(-1, 1, 5),
        "degree": [2, 3, 4, 5, 6],
        "per": [1, 10, 30],
        "l": np.logspace(0, 2, 3),
        "b": [0, ],
        "nu": [1.5, 2.5],
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
        mp = map(lambda pi: "%0.2e %s"
                            % (self.weights[pi],
                               self.name(self.signs[pi], self.args[pi])),
                 range(self.p))
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


def weight_correlation(d1, d2):
    """Measure agreement between two sets of weights in terms of Spearman rho."""
    names = sorted(set(d1.keys()) | set(d2.keys()))
    arr1 = np.zeros((len(names),))
    arr2 = np.zeros((len(names),))
    for i, n in enumerate(names):
        arr1[i] += d1.get(n, 0)
        arr2[i] += d2.get(n, 0)
    return spearmanr(arr1, arr2)


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
    header = ["exp.id", "repl", "P", "n", "rank", "method", "rho", "pvalue", "prec", "recall"]
    names = [nm for nm, _, _ in MultiKernelFunction(1).kernel_generator()]
    header.extend(names)
    writer = csv.DictWriter(open(fname, "w", buffering=0),
                            fieldnames=header, quoting=csv.QUOTE_ALL)
    writer.writeheader()

    # Varying parameters
    range_P = [5, 10, 20]
    range_rank = [3, 5, 10]
    range_n = [100, 300, 1000]

    results = []
    count = 0
    for repl in range(repeats):
        for P, n, rank in it.product(range_P, range_n, range_rank):
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

            # Fit mklaren
            print("Fitting Mklaren ... (%s)" % row)
            try:
                effective_rank = P * rank
                mklaren = mkl.mkl.mklaren.Mklaren(delta=10, rank=effective_rank, lbd=0)
                mklaren.fit(Ks, data["y"])
                mklaren_w = weight_result(names, mklaren.mu)
                mklaren_rho = weight_correlation(mklaren_w, true_w)
                mklaren_pr = weight_PR(true_w, mklaren_w)
            except Exception as e:
                print "Mklaren error", e
                continue
            mklaren_result = {"method": "Mklaren",
                              "rho": mklaren_rho[0], "pvalue": mklaren_rho[1],
                              "prec": mklaren_pr[0], "recall": mklaren_pr[1]}
            mklaren_result.update(row)
            mklaren_result.update(mklaren_w)


            print("Fitting ICD ... (%s)" % row)
            effective_rank = rank
            try:
                icd = mkl.regression.ridge.RidgeLowRank(rank=effective_rank, lbd=0, method="icd")
                icd.fit(Ks, data["y"])
                icd_w = weight_result(names, icd.mu)
                icd_rho = weight_correlation(icd_w, true_w)
                icd_pr = weight_PR(true_w, icd_w)
            except Exception as e:
                print "ICD error", e
                continue
            icd_result = { "method": "ICD",
                            "rho": icd_rho[0], "pvalue": icd_rho[1],
                            "prec": icd_pr[0], "recall": icd_pr[1]}
            icd_result.update(row)
            icd_result.update(icd_w)

            
            print("Fitting CSI ... (%s)" % row)
            effective_rank = rank
            try:
                csi = mkl.regression.ridge.RidgeLowRank(rank=effective_rank, lbd=0,
                                                        method="csi",
                                                        method_init_args={"delta": 10})
                csi.fit(Ks, data["y"])
                csi_w = weight_result(names, csi.mu)
                csi_rho = weight_correlation(csi_w, true_w)
                csi_pr = weight_PR(true_w, csi_w)
            except Exception as e:
                print "CSI error", e
                continue
            csi_result = {"method": "CSI",
                          "rho": csi_rho[0], "pvalue": csi_rho[1],
                          "prec": csi_pr[0], "recall": csi_pr[1]}
            csi_result.update(row)
            csi_result.update(csi_w)

            # Append if all methods go trough
            writer.writerow(true_result)
            writer.writerow(mklaren_result)
            writer.writerow(csi_result)
            writer.writerow(icd_result)
            # results.extend([true_result, mklaren_result, csi_result])

if __name__  == "__main__":
    d = datetime.datetime.now()
    dname = os.path.join("output", "functions", "%d-%d-%d" % (d.year, d.month, d.day))
    if not os.path.exists(dname):
        os.makedirs(dname)
    fname = os.path.join(dname, "results_%d.csv" % len(os.listdir(dname)))
    process(100, fname)


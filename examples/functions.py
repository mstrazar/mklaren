from mklaren.kernel.kernel import exponential_kernel, linear_kernel, poly_kernel, center_kernel, \
    matern_kernel, periodic_kernel, kernel_row_normalize
from scipy.stats import chi2, spearmanr
from sklearn.metrics import precision_score, recall_score
from random import choice
import numpy as np
import mklaren as mkl
import matplotlib.pyplot as plt
import csv


class MultiKernelFunction:

    # Library of function families
    library = {
        "exp": (exponential_kernel, ("gamma", )),
        "lin": (linear_kernel, ()),
        "poly": (poly_kernel, ("p",)),
        # "matern": (matern_kernel, ("nu",)),
        # "periodic": (periodic_kernel, ("p",))
    }

    # Hyper parameter ranges
    values = {
        "gamma": np.logspace(-2, 2, 5),
        "p": [2, 3, 4],
        "b": np.linspace(-3, 3, 5),
        "nu": [2, 3, 4],
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

    def __init__(self, p, center=False, row_normalize=False):
        """
        Initialize a random combination of kernels.
        :param p: Number of kernels.
        """
        assert not (center and row_normalize)
        self.p = p
        self.signs = []
        self.funcs = []
        self.args = []
        self.weights = []
        self.center = center
        self.row_normalize = row_normalize

        for pi in range(p):
            key = choice(self.library.keys())
            self.signs.append(key)

            f, args = self.library[key]
            self.funcs.append(f)

            m = map(lambda a: (a, choice(self.values[a],)), args)
            self.args.append(dict(m))

            # A random kernel weight
            self.weights.append(chi2.rvs(df=1))


    def kernel_matrix(self, X):
        """
        Compute a combined kernel matrix on the fly.
        :param X:
            Data matrix.
        :return:
        """
        n = np.array(X).shape[0]
        K = np.zeros((n, n))
        for pi in range(self.p):
            if self.center:
                K = K + self.weights[pi] * center_kernel(self.funcs[pi](X, X, **self.args[pi]))
            elif self.row_normalize:
                K = K + self.weights[pi] * kernel_row_normalize(self.funcs[pi](X, X, **self.args[pi]))
            else:
                K = K + self.weights[pi] * self.funcs[pi](X, X, **self.args[pi])
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
        am = lambda pi: (self.name(self.signs[pi], self.args[pi]), self.weights[pi])
        mp = map(am, range(self.p))
        return dict(mp)



def generate_kernels(X):
    """
    Generate an Kinterface with all possible kernels.
    :param X:
        Selected data
    :return:
    """

    lib = MultiKernelFunction.library
    vals = MultiKernelFunction.values

    Ks = []
    names = []
    for name, (f, pars) in lib.items():
        if len(pars):
            for p in pars:
                for v in vals[p]:
                    k = kernel_row_normalize(mkl.kernel.kinterface.Kinterface(data=X, kernel=f, kernel_args={p: v})[:, :])
                    Ks.append(k)
                    names.append(MultiKernelFunction.name(name, {p: v}))
        else:
            k = kernel_row_normalize(mkl.kernel.kinterface.Kinterface(data=X, kernel=f)[:, :])
            Ks.append(k)
            names.append(MultiKernelFunction.name(name, {}))
    return Ks, names



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


def process(N=1, P=4, rank=4):
    """
    :param N:
        Number of repetitions.
    :return:
        Simulation results for different methods.


    Ways to compare results in this case:
        - Reconstruct a kernel matrix via regression using all the methods. This is reminiscent of the tru situation,
        where we don't know what sort of relations were used to generate the data.
        - Count the intersection of correctly retrieved kernel families.
        - Measure Spearman rank correlation of obtained versus true weights for appropriate kernels.


    """

    results = []
    for n in range(N):
        data = generate_data(P=P, n=1000, row_normalize=True)
        Ks, names = generate_kernels(data["X"])
        mf = data["mf"]
        true_w = mf.to_dict()

        print("Original kernel")
        print(mf)

        # Fit mklaren

        print("Fitting Mklaren ... (%d)" % n)
        effective_rank = P * rank
        try:
            mklaren = mkl.mkl.mklaren.Mklaren(delta=10, rank=effective_rank, lbd=0)
            mklaren.fit(Ks, data["y"])
        except:
            continue
        mklaren_w = weight_result(names, mklaren.mu)
        mklaren_rho = weight_correlation(mklaren_w, true_w)
        mklaren_pr = weight_PR(true_w, mklaren_w)
        print(mklaren_w)
        print(mklaren_rho)
        print(mklaren_pr)
        print
        mklaren_result = {"N": n, "method": "Mklaren",
            "true": str(mf).replace("\n", ""), "rho": mklaren_rho[0], "pvalue": mklaren_rho[1],
                        "prec": mklaren_pr[0], "recall": mklaren_pr[1]}

        print("Fitting ICD ... (%d)" % n)
        effective_rank = rank
        try:
            csi = mkl.regression.ridge.RidgeLowRank(rank=effective_rank, lbd=0,
                                                method="csi", method_init_args={"delta": 10})
            csi.fit(Ks, data["y"])
        except:
            continue
        csi_w = weight_result(names, csi.mu)
        csi_rho =weight_correlation(csi_w, true_w)
        csi_pr = weight_PR(true_w, csi_w)
        print(csi_w)
        print(csi_rho)
        print(csi_pr)
        print
        csi_result = { "N": n, "method": "CSI",
            "true": str(mf).replace("\n", ""), "rho": csi_rho[0], "pvalue": csi_rho[1],
                         "prec": csi_pr[0], "recall": csi_pr[1]}

        # Append if both methods go trough
        results.append(mklaren_result)
        results.append(csi_result)


    # Write results in a data file
    writer = csv.DictWriter(open("output/functions_center_csi_1000.csv", "w"),
                            fieldnames=results[0].keys(),
                            quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(results)


if __name__  == "__main__":
    process(100)


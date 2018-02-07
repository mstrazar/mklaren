# Console import
from mklaren.util.la import safe_divide as div
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import exponential_kernel, poly_kernel
from numpy import sqrt, array, ones, absolute, sign, hstack, unravel_index, minimum, var, linspace, logspace, arange
from numpy.random import randn
from numpy.linalg import inv, norm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import itertools as it
import csv

hlp = """
    A test Mklaren algorithm based on the Nystrom-type decomposition.
    The basis functions are selected via the LAR criterion.
"""


def norm_matrix(X):
    """ Ensure matrix columns have mean=0, norm=1"""
    return (X - X.mean(axis=0)) / norm(X - X.mean(axis=0), axis=0)


def least_sq(G, y):
    """ Least-squares fit for G, y """
    return G.dot(inv(G.T.dot(G)).dot(G.T).dot(y))


def find_bisector(X):
    """ Find bisector and the normalizing constant for vectors in X."""
    # Compute bisector
    Ga = X.T.dot(X)
    Gai = inv(Ga)
    A = 1.0 / sqrt(Gai.sum())
    omega = A * Gai.dot(ones((X.shape[1], 1)))
    bisector = X.dot(omega)
    assert abs(norm(bisector) - 1) < 1e-3
    return bisector, A


class MklarenNyst:

    def __init__(self, rank, lbd=0, delta=10, debug=False):
        self.rank = rank
        self.lbd = lbd
        self.delta = delta
        self.trained = False
        self.debug = debug
        self.sol_path = []
        self.active = []
        self.G = None
        assert self.lbd >= 0

    def fit(self, Ks, y):
        n = Ks[0].shape[0]

        # Expand full kernel matrices ; basis functions stored in rows
        Xs = array([norm_matrix(K[:, :]).T for K in Ks])

        # Set initial estimate and residual
        regr = ones((n, 1)) * y.mean()
        residual = y - regr
        self.sol_path = [regr]

        # Initial vector is selected by maximum *absolute* correlation
        Cs = Xs.dot(residual)
        q, i, _ = unravel_index(absolute(Cs).argmax(), Cs.shape)
        active = [(q, i)]
        Xa = hstack([sign(Xs[q, i, :].dot(residual)) * Xs[q, i, :].reshape((n, 1)) for q, i in active])

        # Add rank more columns (last step adds two)
        for t in range(1, self.rank):

            # Compute bisector
            bisector, A = find_bisector(Xa)

            # Compute correlations with residual and bisector
            C = max(Xa.T.dot(residual))
            c = Xs.dot(residual)
            a = Xs.dot(bisector)
            assert C > 0

            # Select new basic function and define gradient
            T1 = div((C + c), (A + a))
            T2 = div((C - c), (A - a))
            for q, i in active:
                T1[q, i] = T2[q, i] = float("inf")
            T1[T1 <= 0] = float("inf")
            T2[T2 <= 0] = float("inf")
            T = minimum(T1, T1)
            nq, ni, _ = unravel_index(T.argmin(), T.shape)
            grad = T[nq, ni]

            # Update state
            active = active + [(nq, ni)]
            regr = regr + grad * bisector
            residual = residual - grad * bisector
            self.sol_path += [regr]

            # Update active set
            Xa = hstack([sign(Xs[q, i, :].dot(residual)) * Xs[q, i, :].reshape((n, 1)) for q, i in active])

            # Finish in the last step
            if t == (self.rank - 1):
                # Compute bisector
                bisector, A = find_bisector(Xa)
                C = max(Xa.T.dot(residual))
                grad = C / A
                regr = regr + grad * bisector
                residual = residual - grad * bisector
                self.sol_path += [regr]

        self.active = active
        self.G = hstack([ones((n, 1)) / norm(ones((n, 1))), Xa])

    def fit_greedy(self, Ks, y):
        n = Ks[0].shape[0]

        # Expand full kernel matrices ; basis functions stored in rows
        Xs = array([norm_matrix(K[:, :]).T for K in Ks])
        Xa = ones((n, 1)) / norm(ones((n, 1)))
        regr = least_sq(Xa, y)
        residual = y - regr
        active = []

        for t in range(self.rank):
            # Compute correlations with residual
            c = Xs.dot(residual)

            # Choose a new point to add based on maximum correlation
            for q, i in active:
                c[q, i] = float("-inf")
            nq, ni, _ = unravel_index(absolute(c.argmax()), c.shape)

            # Update active set and model
            active = active + [(nq, ni)]
            Xa = hstack([Xa] + [Xs[nq, ni, :].reshape((n, 1))])
            regr = least_sq(Xa, y)
            residual = y - regr
            self.sol_path += [regr]

        self.active = active
        self.G = Xa

    def predict(self, X):
        pass


def process():

    # Generate data
    n = 100
    rank = 20
    gamma = 0.1
    noise_range = [0, 1.0, 3.0, 10]

    X = linspace(-10, 10, n).reshape((n, 1))
    K = exponential_kernel(X, X, gamma=gamma)
    w = randn(n, 1)
    f = K.dot(w) / K.dot(w).mean()
    noise_vec = randn(n, 1)
    results = dict()

    for noise in noise_range:
        y = f + noise * noise_vec

        # Beacon kernels
        gamma_range = [0.01, 0.03, 0.1, 0.3, 1.0]
        Ks = [Kinterface(data=X,
                         kernel=exponential_kernel,
                         kernel_args={"gamma": gamma}) for gamma in gamma_range]
        model = MklarenNyst(rank=rank)
        model.fit(Ks, y)
        greedy = MklarenNyst(rank=rank)
        greedy.fit_greedy(Ks, y)
        full_path = greedy.sol_path

        # Assert correctness
        assert norm(least_sq(model.G, y) - model.sol_path[-1]) < 1e-5
        assert norm(least_sq(greedy.G, y) - greedy.sol_path[-1]) < 1e-5

        # Residual variance
        evar = lambda fx: (var(y) - var(y - fx)) / var(y)
        sol_var = map(evar, model.sol_path)
        full_var = map(evar, full_path)
        results[noise] = (sol_var, full_var)

    # Residual variance dependent on noise
    plt.figure()
    for noise in noise_range:
        sol_var, full_var = results[noise]
        plt.plot(sol_var, label="LARS", color="orange", linewidth=1+noise/3)
        plt.plot(full_var, label="OLS", color="black", linewidth=1+noise/3)
    plt.xlabel("Step")
    plt.ylabel("Explained variance")
    plt.show()
    plt.grid()

    # Model plot
    for md, col in [(model, "orange"), (greedy, "cyan")]:
        plt.figure(figsize=(12, 4))
        plt.title("Solution path")
        plt.plot(X.ravel(), y.ravel(), "k.")
        for pi, p in enumerate(md.sol_path[:-1]):
            plt.plot(X.ravel(), p.ravel(), "-", color=col, linewidth=1+pi, alpha=0.3)
        plt.plot(X.ravel(), md.sol_path[-1].ravel(), "r-", linewidth=1)
        for pi, (q, i) in enumerate(md.active):
            plt.text(X[i], 0, "%d" % pi)
        plt.plot(X.ravel(), f, "k--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


def test():
    """ A systematic test script. """
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"

    # Generate data ; fixed parameters
    N = 300
    n = 100
    gamma = 0.1
    degree = 2
    X = linspace(-10, 10, n).reshape((n, 1))

    # Varying parameters
    noise_range = [0, 1, 3, 10, 30]
    rank_range = [5, 10, 30]
    p_range = [1, 3, 6]
    kernel_range = [EXPONENTIAL, POLYNOMIAL]

    rows = []

    for kernel, rank, p, noise in it.product(kernel_range, rank_range, p_range, noise_range):

        gamma_range = logspace(-2, 0, p)
        degree_range = arange(p + 1)
        if kernel == EXPONENTIAL:
            K = exponential_kernel(X, X, gamma=gamma)
            Ks = [Kinterface(data=X,
                             kernel=exponential_kernel,
                             kernel_args={"gamma": gam}) for gam in gamma_range]
        elif kernel == POLYNOMIAL:
            K = poly_kernel(X, X, degree=degree)
            Ks = [Kinterface(data=X,
                             kernel=poly_kernel,
                             kernel_args={"degree": deg}) for deg in degree_range]

        for run in range(N):
            w = randn(n, 1)
            f = K.dot(w) / K.dot(w).mean()
            noise_vec = randn(n, 1)
            y = f + noise * noise_vec

            try:
                model = MklarenNyst(rank=rank)
                model.fit(Ks, y)
                greedy = MklarenNyst(rank=rank)
                greedy.fit_greedy(Ks, y)
            except Exception as e:
                print(e)
                pass

            rm = norm(f - model.sol_path[-1])
            rg = norm(f - greedy.sol_path[-1])
            prm = pearsonr(f, model.sol_path[-1])[0][0]
            prg = pearsonr(f, greedy.sol_path[-1])[0][0]

            row = {"N": N, "noise": noise, "method": "lars", "norm": rm, "corr": prm,
                   "kernel": kernel, "p": p, "rank": rank}
            rows.append(row)
            row = {"N": N, "noise": noise, "method": "greedy", "norm": rg, "corr": prg,
                   "kernel": kernel, "p": p, "rank": rank}
            rows.append(row)

        # Write to output
        out_file = "/Users/martins/Dev/mklaren/examples/lars_vs_greedy/results_sys.csv"
        out = open(out_file, "w")
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        out.close()
        print("Written %d rows in %s." % (len(rows), out_file))


if __name__ == "__main__":
    test()

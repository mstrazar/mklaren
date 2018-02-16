# Console import
from mklaren.util.la import safe_divide as div, qr
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import exponential_kernel, poly_kernel, linear_kernel
from numpy import sqrt, array, ones, zeros, argsort, \
    absolute, sign, hstack, unravel_index, minimum, var, linspace, logspace, arange, eye, trace, set_printoptions
from numpy.random import randn, rand
from numpy.linalg import inv, norm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import itertools as it
import csv

hlp = """
    A test Mklaren algorithm based on the Nystrom-type decomposition.
    The basis functions are selected via the LAR criterion.
"""

set_printoptions(precision=2)


def norm_matrix(X):
    """ Ensure matrix columns have mean=0, norm=1"""
    return (X - X.mean(axis=0)) / norm(X - X.mean(axis=0), axis=0)
    # return X / norm(X, axis=0)


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


def bias_variance(X, f, y, noise, lbd=1):
    """ Bias-variance decomposition for design matrix X.
    Although the OLS solution provides non-biased regression estimates, the lower variance solutions produced by
    regularization techniques provide superior MSE performance.
    """
    K = X.dot(X.T)
    n = K.shape[0]
    Ki = inv(K + n * lbd * eye(n, n))
    f_est = K.dot(Ki.dot(y))
    bias2 = n * lbd ** 2 * norm(Ki.dot(f))**2
    variance = float(noise) / n * trace(K.dot(K).dot(Ki).dot(Ki))
    risk = 1.0 / n * norm(f - f_est)**2  # not exactly equal to expected bias/variance
    return sqrt(bias2), variance


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
        self.bias = 0
        assert self.lbd >= 0

    def fit(self, Ks, y):
        n = Ks[0].shape[0]

        # Expand full kernel matrices ; basis functions stored in rows
        Xs = array([norm_matrix(K[:, :]).T for K in Ks])
        # Xs = array([qr(K[:, :])[0].T for K in Ks])

        # Set initial estimate and residual
        self.sol_path = []
        self.bias = y.mean()
        regr = ones((n, 1)) * self.bias
        residual = y - regr

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

        # Last step - full least-sq. solution
        bisector, A = find_bisector(Xa)
        C = max(Xa.T.dot(residual))
        grad = C / A
        regr = regr + grad * bisector

        self.sol_path += [regr]
        self.active = active
        self.G = Xa

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

    def fitted_values(self):
        return self.sol_path[-1]


def test_simple():

    # Generate data
    n = 100
    rank = 5
    gamma = 0.1
    noise_range = [0, 1.0, 3.0, 10, 30]

    X = linspace(-10, 10, n).reshape((n, 1))
    K = exponential_kernel(X, X, gamma=gamma)
    w = randn(n, 1)
    f = K.dot(w) / K.dot(w).mean()
    noise_vec = randn(n, 1)
    results = dict()

    for noise in noise_range:
        y = f + noise * noise_vec
        y -= y.mean()

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

        # Assert correctness; true only for unbiased at the moment
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
            print("Model: %s, step: %d, gamma: %f" % (col, i, gamma_range[q]))
        plt.plot(X.ravel(), f, "k--")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


def test_systematic():
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


def test_bias_variance(noise=3):
    """ Compare selected bandwidths for LARS/stagewise. """
    from collections import Counter

    # Generate data ; fixed parameters
    N = 300
    n = 100
    gamma = 0.1     # True bandwidth
    rank = 10
    gamma_range = [0.01, 0.03, 0.1, 0.3, 1.0]
    R_lars = zeros((N, len(gamma_range)))
    R_stage = zeros((N, len(gamma_range)))

    # Bias variance decomposition
    lbd = 1e-5
    bv_lars = zeros((N, 2))
    bv_stage = zeros((N, 2))
    error_lars = zeros((N,))
    error_stage = zeros((N,))

    for repl in range(N):
        # Generate data
        X = linspace(-10, 10, n).reshape((n, 1))
        K = exponential_kernel(X, X, gamma=gamma)
        # X = rand(n, n)
        # K = linear_kernel(X, X)
        w = randn(n, 1)
        f = K.dot(w) / K.dot(w).mean()
        noise_vec = randn(n, 1)     # Crucial: noise is normally distributed
        y = f + noise * noise_vec

        # Modeling
        Ks = [Kinterface(data=X,
                         kernel=exponential_kernel,
                         kernel_args={"gamma": gam}) for gam in gamma_range]

        lars = MklarenNyst(rank=rank)
        lars.fit(Ks, y)
        stage = MklarenNyst(rank=rank)
        stage.fit_greedy(Ks, y)

        # Count bandwidths
        c_lars = Counter(map(lambda t: t[0], lars.active))
        c_stage = Counter(map(lambda t: t[0], stage.active))
        R_lars[repl, c_lars.keys()] = c_lars.values()
        R_stage[repl, c_stage.keys()] = c_stage.values()

        bv_lars[repl, :] = bias_variance(lars.G, y=y, f=f, noise=noise, lbd=lbd)
        bv_stage[repl, :] = bias_variance(stage.G, y=y, f=f, noise=noise, lbd=lbd)

        error_lars[repl] = norm(f.ravel()-lars.sol_path[-1].ravel())
        error_stage[repl] =norm(f.ravel() - stage.sol_path[-1].ravel())

    # Plot comparison - selected bandwidths
    plt.close("all")
    plt.figure()
    plt.plot(R_lars.sum(axis=0), ".-", label="lars")
    plt.plot(R_stage.sum(axis=0), ".-", label="stagewise")
    plt.legend()
    plt.xticks(range(len(gamma_range)))
    plt.gca().set_xticklabels(gamma_range)
    plt.xlabel("Bandwidth")
    plt.ylabel("Count")

    plt.figure()
    a = min(min(bv_lars[:, 0]), min(bv_stage[:, 0]))
    b = max(max(bv_lars[:, 0]), max(bv_stage[:, 0]))
    plt.plot(bv_lars[:, 0], bv_stage[:, 0], ".")
    plt.plot([a, b], [a, b], "--", color="gray")
    plt.xlabel("Bias (LARS)")
    plt.ylabel("Bias (Stage)")
    plt.show()

    plt.figure()
    a = min(min(bv_lars[:, 1]), min(bv_stage[:, 1]))
    b = max(max(bv_lars[:, 1]), max(bv_stage[:, 1]))
    plt.plot(bv_lars[:, 1], bv_stage[:, 1], ".")
    plt.plot([a, b], [a, b], "--", color="gray")
    plt.xlabel("Variance (LARS)")
    plt.ylabel("Variance (Stage)")
    plt.show()

    plt.figure()
    a = min(min(error_lars), min(error_stage))
    b = max(max(error_stage), max(error_stage))
    plt.plot(error_lars, error_stage, ".")
    plt.plot([a, b], [a, b], "--", color="gray")
    plt.xlabel("Error (LARS)")
    plt.ylabel("Error (Stage)")
    plt.show()


def test_orthog_case(noise=0):
    """ LARS on a constructed orthogonal case"""

    # Fixed data
    n = 3
    X = eye(n, n)
    f = array([[1, -0.48, -0.52]]).T
    noise_vec = randn(n, 1)  # Crucial: noise is normally distributed
    y = f + noise * noise_vec

    # Fit lars
    lars = MklarenNyst(rank=n)
    lars.fit([Kinterface(kernel=linear_kernel, data=X)], y)

    print "\ny"
    print y.ravel()

    print "\nDesign:"
    print lars.G

    print "\n Order statistic"
    inxs = argsort(-absolute(y.ravel()))
    print y[inxs].ravel(), "-"
    print absolute(array(list(y[inxs][1:].ravel())+[0]))
    print "-------------------"
    print absolute(y[inxs]).ravel() - absolute(array(list(y[inxs][1:].ravel())+[0]))

    paths = [zeros((n, 1))] + lars.sol_path
    print "\nSolution transitions:"
    for p0, p1 in zip(paths, paths[1:]):
        print p0.ravel(), "->"
        print p1.ravel(), "grad", (p1-p0).ravel()
        print

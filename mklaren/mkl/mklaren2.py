hlp = """
    A test Mklaren algorithm based on the Nystrom-type decomposition.
    The basis functions are selected via the LAR criterion.
"""

# from ..util.la import safe_divide as div, outer_product, safe_func, qr
# from ..kernel.kinterface import Kinterface
# from ..projection.nystrom import Nystrom

# Console import
from mklaren.util.la import safe_divide as div, outer_product, safe_func, qr
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import center_kernel
from mklaren.projection.nystrom import Nystrom

from numpy import zeros, diag, sqrt, mean, argmax, \
    array, log, eye, ones, absolute, ndarray, where, sign, min as npmin, argmin,\
    sum as npsum, isnan, isinf, hstack, unravel_index, minimum, var
from numpy.linalg import inv, norm
from numpy.random import choice


def norm_matrix(X):
    """ Ensure matrix columns have mean=0, norm=1"""
    return (X - X.mean(axis=0)) / norm(X - X.mean(axis=0), axis=0)


def least_sq(G, y):
    """ Least-squares fit for G, y """
    return G.dot(inv(G.T.dot(G)).dot(G.T).dot(y))


class MklarenNyst:

    def __init__(self, rank, lbd=0, delta=10, debug=False):
        self.rank = rank
        self.lbd = lbd
        self.delta = delta
        self.trained = False
        self.debug = debug
        self.sol_path = []
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

        for t in range(1, self.rank):
            assert abs(norm(Xa[:, t - 1]) - 1) < 1e-8

            # Compute bisector
            Ga = Xa.T.dot(Xa)
            Gai = inv(Ga)
            A = 1.0 / sqrt(Gai.sum())
            omega = A * Gai.dot(ones((len(active), 1)))
            bisector = Xa.dot(omega)
            assert abs(norm(bisector)-1) < 1e-8

            # Compute correlations with residual and bisector
            C = max(Xa.T.dot(residual))
            c = Xs.dot(residual)
            a = Xs.dot(bisector)
            assert C > 0

            # Select new basic function and define gradient
            T1 = div((C + c), (A + a))
            T2 = div((C - c), (A - a))
            for q, i in active:
                T1[q, i] = float("inf")
                T2[q, i] = float("inf")
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
                Ga = Xa.T.dot(Xa)
                Gai = inv(Ga)
                A = 1.0 / sqrt(Gai.sum())
                omega = A * Gai.dot(ones((len(active), 1)))
                bisector = Xa.dot(omega)
                C = max(Xa.T.dot(residual))
                A = 1.0 / sqrt(inv(Xa.T.dot(Xa)).sum())
                grad = C / A
                regr = regr + grad * bisector
                residual = residual - grad * bisector
                self.sol_path += [regr]

        self.active = active
        self.G = hstack([ones((n, 1)) / norm(ones((n, 1))), Xa])

    def predict(self, X):
        pass


if __name__ == "__main__":

    from mklaren.kernel.kernel import exponential_kernel
    from numpy import linspace
    from numpy.random import randn, rand
    import matplotlib.pyplot as plt

    # Generate data
    n = 100
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
        model = MklarenNyst(rank=10)
        model.fit(Ks, y)

        # Least-squares solution
        G = model.G
        lsq = least_sq(G, y)
        print(norm(lsq - model.sol_path[-1]))

        # Full solution path
        full_path = []
        for k in range(len(model.sol_path)):
            Gk = model.G[:, :k+1]
            lsq = least_sq(Gk, y)
            full_path.append(lsq)

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


    # Plot
    plt.figure(figsize=(12, 4))
    plt.title("Solution path")
    plt.plot(X.ravel(), y.ravel(), "k.")
    for pi, p in enumerate(model.sol_path[:-1]):
        plt.plot(X.ravel(), p.ravel(), "c-", linewidth=1+pi, alpha=0.3)
    plt.plot(X.ravel(), lsq.ravel(), "r-", linewidth=1)
    for pi, (q, i) in enumerate(model.active):
        plt.text(X[i], 0, "%d" % pi)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()




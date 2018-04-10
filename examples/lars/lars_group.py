import numpy as np
import matplotlib.pyplot as plt
from mklaren.kernel.kernel import exponential_kernel
from numpy.linalg import norm
from collections import Counter, defaultdict


# Penalty functions
def p_const(p):
    """ Constant (unscaled) penalty. """
    return 1.0


def p_sc(p):
    """ Scaling / diversity penalty. """
    return 1.0 / p if p else 0


def p_ri(p, c=1):
    """ Rich get richer penalty. """
    return c * p / (1.0 + c * p)


def p_act(p, t=10):
    """ Activation function. """
    return max(1, p-t)


def p_sig(p, t=10):
    """ Shifted sigmoid function squeezed between (0.5, 1). """
    return (1 + np.exp(p-t) / (np.exp(p-t) + 1)) / 2


# Explicit defitinions included for debug reasons -- start #
# Cost functions
COSTS = {
    "unscaled": lambda X, y: norm(X.T.dot(y))**2,
    "scaled": lambda X, y: norm(X.T.dot(y))**2 * p_sc(X.shape[1]),
    "rich": lambda X, y: norm(X.T.dot(y))**2 * p_ri(X.shape[1]),
}

# Gain in cost by adding a new vector z orthogonal to X
# Scaled cost can be negative due to inclusion penalty.
GAINS = dict(unscaled=lambda X, y, z: norm(z.T.dot(y)) ** 2,
             scaled=lambda X, y, z: (p_sc(X.shape[1]+1) - p_sc(X.shape[1])) * norm(X.T.dot(y)) ** 2 +
                                     p_sc(X.shape[1]+1) * norm(z.T.dot(y)) ** 2,
             rich=lambda X, y, z:   (p_ri(X.shape[1]+1) - p_ri(X.shape[1])) * norm(X.T.dot(y)) ** 2 +
                                     p_ri(X.shape[1]+1) * norm(z.T.dot(y)) ** 2)

# -- end #

# Signatures
SIGNATURES = {
    "rich": "$f(p) = \\frac{p}{1 + p}$",
    "scaled": "$f(p) = \\frac{1}{p}$",
    "unscaled": "$f(p) = 1$"
}

# Graphical parameters
colors = {"unscaled": "gray",
          "scaled": "pink",
          "rich": "green"}


def group_lars(Xs, y, f=COSTS["unscaled"]):
    """ Group LARS algorithm (full information). """
    cost = np.array([f(x, y) for x in Xs])
    order = np.argsort(-cost)
    pairs = zip(order, order[1:])
    alphas = [1 - np.sqrt(cost[q] / cost[p]) for p, q in pairs] + [1]
    path = [[], [], []]
    mus = []
    costs = []
    for i in range(len(alphas)):
        mu = 0
        for j in order[:i + 1]:
            w = alphas[i] * Xs[j].T.dot(y).ravel()
            mu += Xs[j].dot(w)
            path[j].append(w)
        mus.append(mu)
        costs.append(np.array([f(x, y-mu) for x in Xs]))

    return path, mus, costs


def group_lars_example():
    # Simple MKL data
    n = 100
    X, _, _ = np.linalg.svd(np.random.randn(n, n))
    Xs = [X[:, :10], X[:, 10:15], X[:, 15:18]]
    Ws = [0.1, 1, 0.01]
    y = sum([w * x.dot(np.ones((x.shape[1], 1))) for w, x in zip(Ws, Xs)]).ravel()

    results = dict()
    for cost in COSTS.keys():
        path, mus, costs = group_lars(Xs, y, f=COSTS[cost])
        nb = [norm(mu, ord=2) for mu in mus]
        results[cost] = nb

    plt.figure()
    for ky, nb in results.items():
        plt.plot(nb, ".-", label=ky, color=colors[ky])
    plt.xlabel("Model capacity (kernels) $\\rightarrow$")
    plt.ylabel("$\|\\mu\|_2$")
    plt.legend()
    plt.show()


def group_lars_sampling(Xs, y, rank=10, ky="unscaled"):
    """ Sample individual basis functions from Xs based on gain.
        Return the sampled subspace and the order of selection of kernels. """
    assert rank <= sum([X.shape[1] for X in Xs])
    n = Xs[0].shape[0]
    As = [np.zeros((n, 0)) for X in Xs]
    Cs = dict([(i, set(range(X.shape[1]))) for i, X in enumerate(Xs)])
    P = []
    gains = np.zeros((len(Xs), max(map(lambda x: x.shape[1], Xs))))
    for step in range(rank):
        gains[:, :] = -np.inf
        for i, X in enumerate(Xs):
            for j in Cs[i]:
                gains[i, j] = GAINS[ky](As[i], y, X[:, j])
        i, j = np.unravel_index(np.argmax(gains), gains.shape)
        As[i] = np.hstack((As[i], Xs[i][:, j:j+1]))
        Cs[i].remove(j)
        P.append(i)
    return As, P


def group_lars_sampling_example():
    """ Differences in the order of selected columns for equally correlated data. """
    # Simple MKL data
    noise = 0.01
    n = 100
    t = np.linspace(-10, 10, n).reshape((n, 1))
    K = exponential_kernel(t, t, gamma=0.01)
    X, _, _ = np.linalg.svd(K)
    Xs = [X[:, :8], X[:, 10:15], X[:, 15:18]]
    Ws = [1, 1, 1]
    y = sum([w * x.dot(np.ones((x.shape[1], 1))) for w, x in zip(Ws, Xs)]).ravel() + noise * np.random.randn(n)

    As, P = group_lars_sampling(Xs, y, rank=10, ky="unscaled")
    As, P = group_lars_sampling(Xs, y, rank=10, ky="rich")


def plot_fingerprint(ky="unscaled"):
    """ Plot the order in which kernels are selected for multiple random samples of the data"""

    def remap_indices(P):
        """ Remap indices in P to reflect order of inclusion. """
        d = defaultdict()
        return [d.setdefault(p, len(d)) for p in P]

    noise = 0.01
    N = 100
    n = 300     # Number of data points
    p = 30      # Number of kernels
    rank = int(n / p)

    F = np.zeros((N, p * rank, p))
    for step in range(N):
        X, _, _ = np.linalg.svd(np.random.rand(n, n))
        Xs = [X[:, (j*rank): (1 + j) * rank] for j in range(p)]
        w = np.random.rand(n, 1)
        y = X.dot(w).ravel() + noise * np.random.randn(n)
        As, P = group_lars_sampling(Xs, y, rank=p * rank, ky=ky)
        assert sum(Counter(P).values()) == p * rank
        assert max(Counter(P).values()) <= rank
        inxs = remap_indices(P)
        for i, j in enumerate(inxs):
            F[step, i, j] += 1

    Fm = F.mean(axis=0)
    plt.figure()
    plt.title("Sampling method: %s %s " % (ky, SIGNATURES[ky]))
    plt.imshow(Fm, aspect='auto')
    plt.xlabel("Kernel")
    plt.ylabel("Step")


def plot_cost_functions():
    """ Compare different cost functions and gains. """
    n = 50
    X, _, _ = np.linalg.svd(np.random.randn(n, n))
    y = np.random.randn(n, 1)
    order = np.argsort(-np.absolute(X.T.dot(y)).ravel())
    X = X[:, order]
    gains = dict([(ky, [GAINS[ky](X[:, :i], y, X[:, i]) for i in range(n)])
                  for ky in COSTS.keys()])
    costs = dict([(ky, [COSTS[ky](X[:, :i], y) for i in range(1, n+1)])
                  for ky in COSTS.keys()])

    plt.figure()
    for ky, g in gains.items():
        plt.plot(g, ".-", color=colors[ky], label=ky)
    plt.xlabel("Order")
    plt.ylabel("Gain")
    plt.legend()
    plt.grid()

    plt.figure()
    for ky, c in costs.items():
        plt.plot(c, ".-", color=colors[ky], label=ky)
    plt.xlabel("Order")
    plt.ylabel("Costs")
    plt.legend()
    plt.grid()

    plt.show()


def test_costs_gains():
    """ Test correctness of gain functions for orthogonal data. """
    n = 10
    X, _, _ = np.linalg.svd(np.random.randn(n, n))
    y = np.random.randn(n, 1)
    A = X[:, :9]
    b = X[:, 9:10]
    for ky in COSTS.keys():
        assert abs(COSTS[ky](X, y) - (COSTS[ky](A, y) + GAINS[ky](A, y, b))) < 1e-5


def test_gain_sum():
    """ Total gain is the sum of individual gains. """
    n = 10
    X, _, _ = np.linalg.svd(np.random.randn(n, n))
    y = np.random.randn(n, 1)
    for ky in COSTS.keys():
        total = COSTS[ky](X, y)
        gains = [GAINS[ky](X[:, :i], y, X[:, i]) for i in range(n)]
        assert abs(total - sum(gains)) < 1e-5

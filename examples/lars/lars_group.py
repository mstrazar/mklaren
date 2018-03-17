import numpy as np
import matplotlib.pyplot as plt

COSTS = {
    "unscaled": lambda x, y: np.linalg.norm(x.T.dot(y)),
    "scaled": lambda x, y: np.linalg.norm(x.T.dot(y)) / np.sqrt(x.shape[1]),
    "rich": lambda x, y: np.linalg.norm(x.T.dot(y)) * np.sqrt(x.shape[1] / (1.0 + x.shape[1])),
}

colors = {"unscaled": "gray", "scaled": "pink", "rich": "green"}


def group_lars(Xs, y, f=COSTS["unscaled"]):
    """ Group LARS algorithm (full information). """
    cost = np.array([f(x, y) for x in Xs])
    order = np.argsort(-cost)
    pairs = zip(order, order[1:])
    alphas = [1 - (cost[q] / cost[p]) for p, q in pairs] + [1]
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
        nb = [np.linalg.norm(mu, ord=2) for mu in mus]
        results[cost] = nb

    plt.figure()
    for ky, nb in results.items():
        plt.plot(nb, ".-", label=ky, color=colors[ky])
    plt.xlabel("Model capacity (kernels) $\\rightarrow$")
    plt.ylabel("$\|\\mu\|_2$")
    plt.legend()
    plt.show()

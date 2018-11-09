import numpy as np
import matplotlib.pyplot as plt

class GroupLasso:

    """
    A simple regression model based on the group lasso.
    The design matrix is assumed to be orthonormal.
    """

    def __init__(self, lbd=0.0, max_iter=10):
        self.beta = None
        self.lbd = lbd
        self.bias = 0
        self.max_iter = max_iter

    def fit(self, Xs, y):
        ng = len(Xs)
        ps = np.array([x.shape[1] for x in Xs])
        self.beta = [np.random.rand(p, 1).ravel() for p in ps]
        self.bias = np.mean(y)
        y = y - self.bias

        for t in range(self.max_iter):
            for j in range(ng):
                Sj = Xs[j].T.dot(y - sum([Xs[k].dot(self.beta[k]) for k in range(ng) if k != j]))
                self.beta[j] = max(0, (1 - (self.lbd * np.sqrt(ps[j])) / np.linalg.norm(Sj))) * Sj

    def predict(self, Xs):
        return self.bias + sum(Xs[i].dot(self.beta[i]) for i in range(len(Xs)))

    def df(self, Xs, y):
        """ Compute degrees of freedom based on least-squares coefficients."""
        self.fit(Xs, y)
        X = np.hstack(Xs)
        lsq = np.linalg.solve(X, y)

        # Organize least-squares factors
        lq = []
        i = 0
        for b in self.beta:
            lq.append(lsq[i:i+len(b)])
            i += len(b)

        # Compute df
        df = 0
        for b, q in zip(self.beta, lq):
            bn = np.linalg.norm(b)
            pj = len(b)
            spj = np.sqrt(pj)
            if bn > self.lbd * spj:
                df += 1 + (1 - self.lbd * spj / bn) * (pj - 1)

        return df


def plot_path(Xs, y):
    """ Plot regularization path in slices of a 2D space. """
    X = np.hstack(Xs)
    lsq = np.linalg.solve(X, y)

    # Compute solution path & degrees of freedom
    lbd_range = np.linspace(0, 1, 100)
    W = np.zeros((len(lbd_range), X.shape[1]))
    df = np.zeros((len(lbd_range), ))
    dfk = np.zeros((len(lbd_range),))
    for i, lbd in enumerate(lbd_range):
        model = GroupLasso(lbd=lbd)
        model.fit(Xs, y)
        W[i, :2] = model.beta[0]
        W[i, 2] = model.beta[1]
        dfk[i] = sum(W[i, :] != 0)
        df[i] = model.df(Xs, y)

    # Plot coefficients paths - 2D
    inxs = [[1, 0], [1, 2]]
    fig, ax = plt.subplots(nrows=1, ncols=2)
    for j, inx in enumerate(inxs):
        ax[j].set_xlim((-1.5, 1.5))
        ax[j].set_ylim((-1.5, 1.5))
        ax[j].plot([0, 0], [-1, 1], "k-", linewidth=0.5)
        ax[j].plot([-1, 1], [0, 0], "k-", linewidth=0.5)
        ax[j].set_xlabel("$w_%d$" % inx[0])
        ax[j].set_ylabel("$w_%d$" % inx[1])
        ax[j].plot(W[:, inx][:, 0], W[:, inx][:, 1], "-o", markersize=1, color="gray")
        ax[j].plot(lsq[inx][0], lsq[inx][1], "s", markersize=3, color="red")
    fig.tight_layout()

    # Plot coefficients paths - 1D
    plt.figure()
    for wi, w in enumerate(W.T):
        plt.plot(w)
        plt.text(-1, w[0], "$w_%d$" % wi, horizontalalignment="right")
    plt.gca().set_ylim([-1.2, 1.2])
    plt.gca().set_xlim([-5, W.shape[0]])
    plt.xlabel("Step $\lambda=%d...%d$" % (lbd_range[0], lbd_range[-1]))
    plt.grid()

    # Plot degrees of freedom and simple approximation
    plt.figure()
    plt.plot(df, label="$df(\\beta)$")
    plt.plot(dfk, label="$df = k$")
    plt.legend()
    plt.xlabel("Step $\lambda=%d...%d$" % (lbd_range[0], lbd_range[-1]))
    plt.show()


def test():
    X = np.eye(3)
    Xs = [X[:, :2], X[:, 2:3]]

    noise = 0.01
    w = np.random.randn(3)
    y = X.dot(w) + noise * np.random.randn(3)
    y -= y.mean()

    model = GroupLasso(lbd=0.2)
    model.fit(Xs, y)
    yp = model.predict(Xs)

    plot_path(Xs, y)
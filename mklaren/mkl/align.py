from ..kernel.kernel import center_kernel
from ..util.la import fro_prod, fro_prod_low_rank
from numpy import zeros, sqrt
from numpy import ndarray


class Align:
    """
    C. Cortes, M. Mohri, and A. Rostamizadeh,
    "Algorithms for Learning Kernels Based on Centered Alignment,"
    J. Mach. Learn. Res., vol. 13, pp. 795-828, Mar. 2012.

    3.1 Independent-alignment based algorithm (align).

    Independent-alignment based algorithm (align).
    Each kernel is aligned to the ideal kernel independently.
    The values of alignments for each individual kernel are kernel weights.

    Model attributes:
            Kappa optimally alignme kernel matrix.
            mu Weights for individual kernels
    """


    def __init__(self, q=2):
        """
        :param q:
            q-norm of kernel weights (int, default=2).
        """
        self.q = 2


    def fit(self, Ks, y, holdout=None):
        """

        :param Ks:
            List of kernel matrices.
        :param y:
            Class labels y_i \in {-1, 1}.
        :param holdout:
        :return:
            Kappa:
                aligned kernel.
            mu:
                Kernel weights.
        """
        q = self.q
        m = len(y)
        mu = zeros((len(Ks), ))
        y = y.reshape((m, 1))

        # Generalization to Kinterfaces
        Ks = [K[:, :] for K in Ks]

        # Filter out hold out values
        if not isinstance(holdout, type(None)):
            holdin = sorted(list(set(range(m)) - set(holdout)))
            y = y[holdin]
            Ksa = map(lambda k: k[holdin, :][:, holdin], Ks)
            en = enumerate(Ksa)
            Ky = y.dot(y.T)
        else:
            Ksa = Ks
            en = enumerate(Ksa)
            Ky = y.dot(y.T)

        for ki, K in en:
            mu[ki] = fro_prod(center_kernel(K), Ky)**(1.0/(q - 1))
        mu = mu / mu.sum()
        Kappa = sum([mu_i * k_i for mu_i, k_i in zip(mu, Ks)])

        self.Kappa   = Kappa
        self.mu      = mu
        self.trained = True


    def __call__(self, i, j):
        assert self.trained
        if isinstance(i, ndarray):
            i = i.astype(int).ravel()
        if isinstance(j, ndarray):
            j = j.astype(int).ravel()
        if isinstance(i, int) and isinstance(j, int):
            return self.Kappa[i, j]
        else:
            return self.Kappa[i, :][:, j]

    def __getitem__(self, item):
        assert self.trained
        return self.Kappa[item]



class AlignLowRank(Align):
    """
    Use the align method using low-rank kernels.
    Useful for computing alignment of low-rank representations.

    """

    def fit(self, Gs, y, holdout=None):
        """

        :param Gs:
            Low-rank kernel representations.
            Must be equal to matrices.
        :param y:
            Target vector.
        :param holdout:
            Holdout test set.
        """
        q   = self.q
        m   = len(y)
        mu  = zeros((len(Gs), ))
        y   = y.reshape((m, 1))

        # Filter out hold out values
        if not isinstance(holdout, type(None)):
            holdin = sorted(list(set(range(m)) - set(holdout)))
            y      = y[holdin]
            Gsa    = map(lambda g: g[holdin, :], Gs)
            en     = enumerate(Gsa)
        else:
            Gsa    = Gs
            en     = enumerate(Gsa)

        for gi, G in en:
            Gc     = G - G.mean(axis=0)
            mu[gi] = fro_prod_low_rank(Gc, y)**(1.0/(q - 1))

        mu       = mu / mu.sum()

        self.Gs      = Gs
        self.mu      = mu
        self.trained = True


    def __call__(self, i, j):
        assert self.trained
        if isinstance(i, ndarray):
            i = i.astype(int).ravel()
        if isinstance(j, ndarray):
            j = j.astype(int).ravel()

        return sum([m * (G[i, :].dot(G[j, :].T))
                    for m, G in zip(self.mu, self.Gs)])


    def __getitem__(self, item):
        assert self.trained
        return sum([m * (G[item[0]].dot(G[item[1]].T))
                    for m, G in zip(self.mu, self.Gs)])

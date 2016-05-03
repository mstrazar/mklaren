from ..util.la import fro_prod, fro_prod_low_rank
from ..kernel.kernel import center_kernel, center_kernel_low_rank
from numpy import zeros, eye, array, ndarray
from numpy.linalg import inv, norm
from itertools import combinations
from cvxopt.solvers import qp as QP, options
from cvxopt import matrix
options['show_progress'] = False


class Alignf:

    """
    C. Cortes, M. Mohri, and A. Rostamizadeh,
    "Algorithms for Learning Kernels Based on Centered Alignment,"
    J. Mach. Learn. Res., vol. 13, pp. 795-828, Mar. 2012.

    3.2 Alignment maximization algorithm (alignf).

    Model attributes:
        Kappa optimally alignme kernel matrix.
        mu Weights for individual kernels
    """

    def __init__(self, typ="linear"):
        assert typ in ["linear", "convex"]
        self.typ = typ
        self.trained = False
        self.low_rank = False


    def fit(self, Ks, y, holdout=None):
        """
        :param Ks:
            List of kernel matrices.
        :param y:
            Class labels y_i \in {-1, 1}.
        """
        m = len(y)
        p = len(Ks)
        y = y.reshape((m, 1))

        # Generalization to Kinterfaces
        Ks = [K[:, :] for K in Ks]

        # Filter out hold out values
        if not self.low_rank:
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
        else:
            if not isinstance(holdout, type(None)):
                holdin = sorted(list(set(range(m)) - set(holdout)))
                y      = y[holdin]
                Ksa    = map(lambda k: k[holdin, :], Ks)
                en     = enumerate(Ksa)
            else:
                Ksa    = Ks
                en     = enumerate(Ksa)



        a = zeros((p, 1))
        M = zeros((p, p))

        if not self.low_rank:
            for (i, K), (j, L) in combinations(list(en), 2):
                M[i, j] = M[j, i] = fro_prod(center_kernel(K), center_kernel(L))
                if a[i] == 0:
                    M[i, i] = fro_prod(center_kernel(K), center_kernel(K))
                    a[i] = fro_prod(center_kernel(K), Ky)
                if a[j] == 0:
                    M[j, j] = fro_prod(center_kernel(L), center_kernel(L))
                    a[j] = fro_prod(center_kernel(L), Ky)
        else:
            for (i, K), (j, L) in combinations(list(en), 2):
                M[i, j] = M[j, i] = fro_prod_low_rank(center_kernel_low_rank(K),
                                                      center_kernel_low_rank(L))
                if a[i] == 0:
                    M[i, i] = fro_prod_low_rank(center_kernel_low_rank(K),
                                                center_kernel_low_rank(K))
                    a[i] = fro_prod_low_rank(center_kernel_low_rank(K), y)
                if a[j] == 0:
                    M[j, j] = fro_prod_low_rank(center_kernel_low_rank(L),
                                                center_kernel_low_rank(L))
                    a[j] = fro_prod_low_rank(center_kernel_low_rank(L), y)


        if self.typ == "linear":
            Mi = inv(M)
            mu = Mi.dot(a) / norm(Mi.dot(a), ord=2)
        elif self.typ == "convex":
            Q = matrix(M)
            r = matrix(-2 * a.ravel())
            G = -1 * matrix(eye(p, p))
            h = matrix(0.0, (p, 1))
            sol = QP(Q, r, G, h)
            mu = array(sol["x"]).ravel()
            mu = mu / norm(mu, ord=1)


        if not self.low_rank:
            Kappa = sum([mu_i * center_kernel(k_i) for mu_i, k_i in zip(mu, Ks)])
            self.Kappa = Kappa
        else:
            self.Gs = map(lambda g: center_kernel_low_rank(g), Ks)

        mu = mu.ravel()
        self.mu = mu
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



class AlignfLowRank(Alignf):
    """
    Use the align method using low-rank kernels.
    Useful for computing alignment of low-rank representations.

    """
    def __init__(self, typ="linear"):
        assert typ in ["linear", "convex"]
        self.typ     = typ
        self.trained = False
        self.low_rank = True


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


"""
The algorithms based on centered aligmnent proposed in

C. Cortes, M. Mohri, and A. Rostamizadeh, "Algorithms for Learning Kernels Based on Centered Alignment," J. Mach. Learn. Res., vol. 13, pp. 795-828, Mar. 2012.

Given :math:`p` kernel matrices :math:`\mathbf{K}_1, \mathbf{K}_2, ..., \mathbf{K}_p`, centered kernel alignment learns a linear combination of kernels resulting in a combined kernel matrix.

.. math::
    \mathbf{K}_{c\mu} = \sum_{q=1}^p \mu_q \mathbf{K}_{cq}

where :math:`\mathbf{K}_{cq}` is the centered kernel matrix.

.. math::
    \mathbf{K}_{cq} = (\mathbf{I} - \dfrac{\mathbf{11}^T}{n})\mathbf{K}_q (\mathbf{I} - \dfrac{\mathbf{11}^T}{n})

The ``Alignf`` method optimies with respect to :math:`\mathbf{\mu}` to maximize centered alignment.

.. math::
    max_{\mathbf{\mu}} \dfrac{<\mathbf{K}_{c\mu}, \mathbf{y}\mathbf{y}^T>_F} {n <\mathbf{K}_{c\mu}, \mathbf{K}_{c\mu}>_F}

such that (``typ=linear``):

.. math::
    \sum \mu_q = 1

or contraining sum of weights to a convex combination (``typ=convex``):

.. math::
    \sum \mu_q = 1,  \mu_q > 0, q = 1...p
"""

from ..util.la import fro_prod, fro_prod_low_rank
from ..kernel.kernel import center_kernel, center_kernel_low_rank
from numpy import zeros, eye, array, ndarray, ones
from numpy.linalg import inv, norm
from itertools import combinations
from cvxopt.solvers import qp as QP, options
from cvxopt import matrix
options['show_progress'] = False


class Alignf:

    def __init__(self, typ="linear", lbd2=1e-5):
        """
        :param typ: (``str``) "linear" or "convex" combination of kernels.
        :param lbd2: (``float``) Noise (regularization) to guarantee inversion of the kernel-target projection matrix.
        """
        assert typ in ["linear", "convex"]
        self.typ = typ
        self.trained = False
        self.low_rank = False
        self.lbd2 = lbd2


    def fit(self, Ks, y, holdout=None):
        """
        Learn weights for kernel matrices or Kinterfaces.

        :param Ks: (``list``) of (``numpy.ndarray``) or of (``Kinterface``) to be aligned.

        :param y: (``numpy.ndarray``) Class labels :math:`y_i \in {-1, 1}` or regression targets.

        :param holdout: (``list``) List of indices to exlude from alignment.
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

        if p == 1:
            a = ones((p, 1))
            M = ones((p, p))
        else:
            a = zeros((p, 1))
            M = zeros((p, p))
            if not self.low_rank:
                for (i, K), (j, L) in combinations(list(en), 2):
                    M[i, j] = M[j, i] = fro_prod(center_kernel(K), center_kernel(L))
                    if a[i] == 0:
                        M[i, i] = self.lbd2 + fro_prod(center_kernel(K), center_kernel(K))
                        a[i] = fro_prod(center_kernel(K), Ky)
                    if a[j] == 0:
                        M[j, j] = self.lbd2 + fro_prod(center_kernel(L), center_kernel(L))
                        a[j] = fro_prod(center_kernel(L), Ky)
            else:
                for (i, K), (j, L) in combinations(list(en), 2):
                    M[i, j] = M[j, i] = fro_prod_low_rank(center_kernel_low_rank(K),
                                                          center_kernel_low_rank(L))
                    if a[i] == 0:
                        M[i, i] = self.lbd2 + fro_prod_low_rank(center_kernel_low_rank(K),
                                                    center_kernel_low_rank(K))
                        a[i] = fro_prod_low_rank(center_kernel_low_rank(K), y)
                    if a[j] == 0:
                        M[j, j] = self.lbd2 + fro_prod_low_rank(center_kernel_low_rank(L),
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
        """
        Access portions of the combined kernel matrix at indices i, j.

        :param i: (``int``) or (``numpy.ndarray``) Index/indices of data points(s).

        :param j: (``int``) or (``numpy.ndarray``) Index/indices of data points(s).

        :return:  (``numpy.ndarray``) Value of the kernel matrix for i, j.
        """
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
        """
        Access portions of the kernel matrix generated by ``kernel``.

        :param item: (``tuple``) pair of: indices or list of indices or (``numpy.ndarray``) or (``slice``) to address portions of the kernel matrix.

        :return:  (``numpy.ndarray``) Value of the kernel matrix for item.
        """
        assert self.trained
        return self.Kappa[item]



class AlignfLowRank(Alignf):
    """
    Use the align method using low-rank kernels.
    Useful for computing alignment of low-rank representations.
    """
    def __init__(self, typ="linear", lbd2=1e-5):
        """
        :param typ: (``str``) "linear" or "convex" combination of kernels.
        """
        assert typ in ["linear", "convex"]
        self.typ     = typ
        self.trained = False
        self.low_rank = True
        self.lbd2 = lbd2

    def __call__(self, i, j):
        """
        Access portions of the combined kernel matrix at indices i, j.

        :param i: (``int``) or (``numpy.ndarray``) Index/indices of data points(s).

        :param j: (``int``) or (``numpy.ndarray``) Index/indices of data points(s).

        :return:  (``numpy.ndarray``) Value of the kernel matrix for i, j.
        """
        assert self.trained
        if isinstance(i, ndarray):
            i = i.astype(int).ravel()
        if isinstance(j, ndarray):
            j = j.astype(int).ravel()
        return sum([m * (G[i, :].dot(G[j, :].T))
                    for m, G in zip(self.mu, self.Gs)])


    def __getitem__(self, item):
        """
        Access portions of the kernel matrix generated by ``kernel``.

        :param item: (``tuple``) pair of: indices or list of indices or (``numpy.ndarray``) or (``slice``) to address portions of the kernel matrix.

        :return:  (``numpy.ndarray``) Value of the kernel matrix for item.
        """
        assert self.trained
        return sum([m * (G[item[0]].dot(G[item[1]].T))
                    for m, G in zip(self.mu, self.Gs)])


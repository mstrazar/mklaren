from ..kernel.kernel import center_kernel, center_kernel_low_rank
from ..kernel.projection.csi import CSI
from numpy import zeros, sum as npsum, ndarray, hstack
from numpy.linalg import norm, inv
from itertools import combinations



class AlignCSI:
    """
        Perform CSI independently for each kernel and merge kernels via centered alignment.
        Model with ICD is recovered if parameter is set lbd=0.

        Model attributes:
            Kappa optimally alignme kernel matrix.
            mu Weights for individual kernels
            Gs Cholesky factors.
            Qs Orthonormal Q matrices.
            Rs Upper-triangular matrices.
            M  Pair-wise kernel Frobenius product.
            a  Frobenius product kernel vs. class vector.
    """


    def __init__(self, **kwargs):
        """
        :param kwargs:
            Parameters for inmf.kernel.low_rank.csi method.
        """
        self.csi_args = kwargs
        self.trained = False


    def fit(self, Ks, y, holdout=None):
        """
        :param Ks:
            List of kernel matrices.
        :param y:
            Class labels y_i \in {-1, 1}.

        Note that after CSI model, the order of rows must be restored due
        to permutations used. There are already restored as a part of the CSI
        mehtod in this case.
        """
        model = CSI(**self.csi_args)
        p = len(Ks)
        Gs = []
        Qs = []
        Rs = []
        for K in Ks:
            model.fit(K, y, holdout)
            Gs.append(model.G)
            Rs.append(model.R)
            Qs.append(model.Q)

        # Construct holdin set if doing transductive learning
        holdin = None
        if holdout is not None:
            n = Ks[0].shape[0]
            holdin = list(set(range(n)) - set(holdout))

        # Solve for the best linear combination of weights
        a = zeros((p, 1))
        M = zeros((p, p))
        for (i, Gu), (j, Hu) in combinations(enumerate(list(Gs)), 2):
            G = center_kernel_low_rank(Gu)
            H = center_kernel_low_rank(Hu)
            M[i, j] = M[j, i] = npsum(G.T.dot(H)**2)
            if a[i] == 0:
                M[i, i] = npsum(G.T.dot(G)**2)
                if holdin is None:
                    a[i] = npsum(G.T.dot(y)**2)
                else:
                    a[i] = npsum(G[holdin, :].T.dot(y[holdin])**2)
            if a[j] == 0:
                M[j, j] = npsum(H.T.dot(H)**2)
                if holdin is None:
                    a[j] = npsum(H.T.dot(y)**2)
                else:
                    a[j] = npsum(H[holdin, :].T.dot(y[holdin])**2)

        Mi = inv(M)
        mu = Mi.dot(a) / norm(Mi.dot(a), ord=2)
        self.Gs = map(center_kernel_low_rank, Gs)
        self.G  = hstack(Gs)
        self.mu = mu
        self.trained = True


    def __call__(self, i, j):
        assert self.trained
        if isinstance(i, ndarray):
            i = i.astype(int).ravel()
        if isinstance(j, ndarray):
            j = j.astype(int).ravel()
        return sum([self.mu[gi] * G[i, :].dot(G[j, :].T)
                    for gi, G in enumerate(self.Gs)])


    def __getitem__(self, item):
        assert self.trained
        return sum([self.mu[gi] * G[item[0], :].dot(G[item[1], :].T)
                    for gi, G in enumerate(self.Gs)])



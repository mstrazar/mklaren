from sklearn.kernel_approximation import RBFSampler
from numpy import ndarray, zeros, isnan, isinf, where, absolute, eye, sqrt, cos, sin, hstack, ceil, floor
from numpy.linalg import norm
from numpy.random import seed
from scipy.stats import multivariate_normal as mvn


def exponential_density(rank, d, sigma=None, gamma=1):
    """ Return samples for the exponential kernel. """
    if sigma is None:
        sigma = sqrt(2 * d * gamma)
    return mvn.rvs(mean=zeros(d,), cov=sigma**2 * eye(d, d), size=rank).reshape((rank, d))


class RFF:
    """
    Generalized Random Fourier Features Sampler.
    """

    def __init__(self, d, n_components=10, density=exponential_density, random_state=None, **kwargs):
        self.d = d
        self.rank = n_components
        self.W = None
        self.b = None
        self.density = density
        self.kwargs = kwargs
        if random_state is not None:
            seed(random_state)

    def fit(self):
        """ Sample random directions for a given kernel. """
        self.W = self.density(self.rank, self.d, **self.kwargs)

    def transform(self, X):
        """ Map X to approximation of the kernel. """
        # Rahimi & Recht 2007 - real part only (rank)
        # Ton et. al 2018 - real + complex (2k)
        n, d = X.shape
        A = X.dot(self.W.T)
        j_cos = int(ceil(self.rank/2.0))
        j_sin = int(floor(self.rank/2.0))
        assert j_cos + j_sin == self.rank
        return sqrt(2.0 / self.rank) * hstack((cos(A[:, :j_cos]), sin(A[:, :j_sin]))).reshape((n, self.rank))

    def fit_transform(self, X):
        """ Map X to approximation of the kernel. """
        # Rahimi & Recht 2007 - real part only (rank)
        # Ton et. al 2018 - real + complex (2k)
        n, self.d = X.shape
        self.fit()
        return self.transform(X)


class RFF_NS:
    """
    Generalized Random Fourier Features Sampler for non-stationary kernel.
    """

    def __init__(self, d, n_components=10, random_state=None,
                 density1=exponential_density, kwargs1={},
                 density2=exponential_density, kwargs2={}):
        self.d = d
        self.rank = n_components
        self.W1 = None
        self.W2 = None
        self.density1 = density1
        self.density2 = density2
        self.kwargs1 = kwargs1
        self.kwargs2 = kwargs2
        if random_state is not None:
            seed(random_state)

    def fit(self):
        """ Sample random directions for a given kernel. """
        self.W1 = self.density1(self.rank, self.d, **self.kwargs1)
        self.W2 = self.density2(self.rank, self.d, **self.kwargs2)

    def transform(self, X):
        """ Map X to approximation of the kernel. """
        # Ton et. al 2018 - real + complex (2k)
        n, d = X.shape
        A = X.dot(self.W1.T)
        B = X.dot(self.W2.T)
        j_cos = int(ceil(self.rank / 2.0))
        j_sin = int(floor(self.rank / 2.0))
        assert j_cos + j_sin == self.rank
        return sqrt(1.0 / (2 * self.rank)) * hstack((cos(A[:, :j_cos]) + cos(B[:, :j_cos]),
                                                     sin(A[:, :j_sin]) + sin(B[:, :j_sin]))).reshape((n, self.rank))

    def fit_transform(self, X):
        """ Map X to approximation of the kernel. """
        # Ton et. al 2018 - real + complex (2k)
        n, self.d = X.shape
        self.fit()
        return self.transform(X)


# Constants
RFF_TYP_STAT = "rff"
RFF_TYP_NS = "rff_nonstat"
RFF_TYPES = (RFF_TYP_STAT, RFF_TYP_NS)


class RFF_KMP:
    """
        Random Fourier features (Rahimi & Recht 2007).
        Approximation of the exponentiated quadratic kernel matrix on the given data sample.

        Extends scikit-learn RBFSampler and upgrades with matching pursuit to select features
        based on a target vector. No kernels are provided this time, as they are implicit
        with the selected Fourier transform.

        Stores a transformed matrix G to be used with Ridge Regression.

        Look-ahead columns (delta) are employed to select new features in a similar fashion
        to CSI/Mklaren.

        To employ L2-regularization, use the basic least-sq. algorithm, but augment the data
        as in LARS.

        The `transform` method is used to compute features for new data given a set of basis
        functions.

        Supervised feature construction is described here:
            1. A. Rahimi, B. Recht, in Advances in Neural Information Processing Systems
            (NIPS) (2009), vol. 1, pp. 1177-1184.
    """

    def __init__(self, rank, delta, lbd=0, gamma_range=(1.0,), random_state=None, normalize=False,
                 typ=RFF_TYP_STAT):
        """
        :param rank:
            Rank of approximation.
        :param delta:
            Number of lookahead columns.
            With delta=0, random selection of features is recovered.
        :param gamma_range:
            Hyperparameter to RBF kernels to be modeled
        :param random_state:
            Random state.
        :param normalize:
            Normalize the implicit feature space.
        """
        assert isinstance(gamma_range, list) or isinstance(gamma_range, ndarray)

        self.G = None
        self.active_set = None
        self.beta = None
        self.bias = None
        self.gmeans = None
        self.gnorms = None

        self.gamma_range = gamma_range
        self.random_state = random_state
        self.rank = rank
        self.delta = delta
        self.lbd = lbd
        self.normalize = normalize
        self.typ = typ
        self.samplers = None

    def fit(self, X, y):
        """
        Select random features based on the target vector. Employ KMP to select kernels
        from different bandwidth.

        Store transformed data inside the object.

        :param X:
            Data in original feature space.
        :param y:
            Target vector.
        """
        n, d = X.shape
        p = self.rank + self.delta

        # Initialize samples
        self.samplers = [RFF(d=d,
                             n_components=self.delta + self.rank,
                             random_state=self.random_state,
                             gamma = g)
                         if self.typ == RFF_TYP_STAT
                         else RFF_NS(d=d,
                                     n_components=self.delta + self.rank,
                                     random_state=self.random_state,
                                     kwargs1={"gamma": g},
                                     kwargs2={"gamma": g})
                         for g in self.gamma_range]

        # Target signal
        self.bias = y.mean()
        residual = zeros((n + p, 1))
        residual[:n] = y.reshape((n, 1)) - self.bias

        # Kernel index / feature index
        self.G = zeros((n, self.rank))
        self.beta = zeros((self.rank,))
        self.active_set = zeros((self.rank, 2), dtype=int)

        # Store all candidate features generated by self.samplers
        # Added L-2 regularization in the normalized feature space
        Rff = zeros((len(self.samplers), n + p, p))
        self.gmeans = zeros((len(self.samplers), p))
        self.gnorms = zeros((len(self.samplers), p))
        for si, samp in enumerate(self.samplers):
            Xt = samp.fit_transform(X)
            self.gmeans[si] = Xt.mean(axis=0).reshape((1, p))
            self.gnorms[si] = norm(Xt - self.gmeans[si], axis=0).reshape((1, p))
            if self.normalize:
                Rff[si, :n, :] = sqrt(1 + self.lbd) * (Xt - self.gmeans[si]) / self.gnorms[si]
                Rff[si, n:, :] = sqrt(1 + self.lbd) * sqrt(self.lbd) * eye(p, p)
            else:
                Rff[si, :n, :] = sqrt(1 + self.lbd) * Xt
                Rff[si, n:, :] = sqrt(1 + self.lbd) * sqrt(self.lbd) * eye(p, p)

        # Compute costs and select best features
        costs = zeros((len(self.samplers), p))
        for ri in xrange(self.rank):

            for si, rf in enumerate(Rff):
                norms = norm(rf, axis=0).reshape((1, p))
                projections = rf.T.dot(residual).reshape((1, p))
                costs[si] = absolute(projections / norms)

            costs[isnan(costs)] = 0
            costs[isinf(costs)] = 0

            # Select active column
            inxs = where(costs == costs.max())
            si, col = int(inxs[0][0]), int(inxs[1][0])

            # Store projection and update residual
            gj = Rff[si][:, col].reshape((n + p, 1))
            self.active_set[ri, :] = [si, col]
            self.beta[ri] = (gj.T.dot(residual)) / norm(gj) ** 2
            self.G[:, ri] = gj.ravel()[:n]

            residual = residual - self.beta[ri] * gj

    def transform(self, X):
        """
        Map new samples in X to the preselected set of features.

        :param X:
            Data in original feature space.
        :return:
            Transformed data
        """
        nt = X.shape[0]
        p = self.rank + self.delta
        Gt = zeros((nt, self.rank))

        # Compute projections in pre-trained self.samplers
        Rff = zeros((len(self.samplers), nt, p))
        for si, samp in enumerate(self.samplers):
            if si not in self.active_set[:, 0]:
                continue
            if self.normalize:
                Rff[si] = (samp.transform(X) - self.gmeans[si]) / self.gnorms[si]
            else:
                Rff[si] = samp.transform(X)

        # Return matching columns
        for ri in xrange(self.rank):
            si, col = self.active_set[ri, :]
            Gt[:, ri] = Rff[si][:, col]
        return Gt

    def predict(self, X):
        """
        Predict the signal for new samples.

        :param X:
            Data in original feature space.
        :return:
            Transformed data
        """
        Gt = self.transform(X)
        return self.bias + Gt.dot(self.beta).ravel()

""" Methods related to calculation of kernel function values and kernel
    matrices.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.gaussian_process.kernels import Matern
from scipy.spatial.distance import cdist

# Install the GPy module to use included kernels
try:
    import GPy
except ImportError:
    pass


def correct_xy(x, y):
    """
    Convert matrices to dense and correct shapes.

    :param x: (``numpy.ndarray``) 2D or 1D array

    :param y: (``numpy.ndarray``) 2D or 1D array

    :return:  (``numpy.ndarray``) Convert x, y to dense, 2D arrays.
    """
    if sp.isspmatrix(x) and sp.isspmatrix(y):
        x = np.array(x.todense())
        y = np.array(y.todense())
    if not hasattr(x, "shape") or np.asarray(x).ndim == 0:
        x = np.reshape(np.array([x]), (1, 1))
    if not hasattr(y, "shape") or np.asarray(y).ndim == 0:
        y = np.reshape(np.array([y]), (1, 1))

    if np.asarray(x).ndim == 1: x = np.reshape(np.array([x]), (len(x), 1))
    if np.asarray(y).ndim == 1: y = np.reshape(np.array([y]), (len(y), 1))
    return x, y


def linear_kernel(x, y, b=0):
        """
        The linear kernel (the usual dot product in n-dimensional space).

        .. math::
            k(\mathbf{x}, \mathbf{y}) = b + \mathbf{x}^T \mathbf{y}

        :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param b: (``float``) Bias term.

        :return: (``numpy.ndarray``) Kernel value/matrix between data points.
        """
        if isinstance(x, int):
            return x * y
        if sp.isspmatrix(x):
            return b + np.array(x.dot(y.T).todense())
        else:
            return b + x.dot(y.T)


def linear_kernel_noise(x, y, b=1, noise=1):
    """
    The linear kernel (the usual dot product in n-dimensional space). A noise term is
    added explicitly to avoid singular kernel matrices

    .. math::
        k(\mathbf{x}, \mathbf{y}) = b + \mathbf{x}^T \mathbf{y} + noise \cdot (\mathbf{x} == \mathbf{y})

    :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

    :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

    :param b: (``float``) Bias term.

    :param noise: (``float``) Noise term.

    :return: (``numpy.ndarray``) Kernel value/matrix between data points.
    """
    D = cdist(x, y, metric="euclidean")
    if isinstance(x, int):
        return x * y
    if sp.isspmatrix(x):
        return b + np.array(x.dot(y.T).todense()) + noise * (D == 0)
    else:
        return b + x.dot(y.T) + noise * (D == 0)


def poly_kernel(x, y, degree=2, b=0):
        """
        The polynomial kernel.

        .. math::
            k(\mathbf{x}, \mathbf{y}) = (b + \mathbf{x}^T \mathbf{y})^p

        :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param degree: (``float``) Polynomial degree.

        :param b: (``float``) Bias term.

        :return: (``numpy.ndarray``) Kernel value/matrix between data points.
        """
        if sp.isspmatrix(x):
            return np.array(x.dot(y.T).todense()) ** degree
        if not hasattr(x, "shape"):
            return (b + x * y) ** degree
        else:
            return (b + x.dot(y.T)) ** degree


def sigmoid_kernel(x, y, c=1, b=0):
        """
        The sigmoid kernel.

        .. math::
            k(\mathbf{x}, \mathbf{y}) = tan(c \mathbf{x}^T \mathbf{y} + b)

        :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param c: (``float``) Scale.

        :param b: (``float``) Bias.

        :return: (``numpy.ndarray``) Kernel value/matrix between data points.
        """
        if sp.isspmatrix(x) and sp.isspmatrix(y):
            x = np.array(x.todense())
            y = np.array(y.todense())
        if not hasattr(x, "shape"):
            return np.tanh(c * x * y + b)
        else:
            return np.tanh(c * x.dot(y.T) + b)


def exponential_kernel(x, y, sigma=2.0, gamma=None):
    """
    The exponential quadratic / radial basis kernel (RBF) kernel.

        .. math::
            k(\mathbf{x}, \mathbf{y}) = exp\{\dfrac{\|\mathbf{x} - \mathbf{y}\|^2}{\sigma^2} \}

        or

        .. math::
            k(\mathbf{x}, \mathbf{y}) = exp\{\gamma \|\mathbf{x} - \mathbf{y}\|^2 \}

        :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param sigma: (``float``) Length scale.

        :param gamma: (``float``) Scale.

        :return: (``numpy.ndarray``) Kernel value/matrix between data points.
    """

    if gamma is None:
        gamma = 1.0 / (2.0 * sigma ** 2)

    if sp.isspmatrix(x) and sp.isspmatrix(y):
        x = np.array(x.todense())
        y = np.array(y.todense())
    if not hasattr(x, "shape"):
        return np.exp(-gamma  * np.linalg.norm(x - y, ord=2)**2)
    if np.asarray(x).ndim == 0:
        return np.exp(-gamma  * (x - y)**2)
    if len(x.shape) >= 2 or len(y.shape) >= 2:
        return np.exp(-gamma * cdist(x, y, metric="euclidean")**2)
    return np.exp(-gamma * np.linalg.norm(x - y, ord=2)**2)


def exponential_cosine_kernel(x, y, gamma=1, omega=1):
    """
    A sum of exponential quadratic and a cosine kernel.

        .. math::
            d = \|\mathbf{x} - \mathbf{y}\|
        .. math::
            k(\mathbf{x}, \mathbf{y}) = \dfrac{1}{2} exp\{\dfrac{d^2}{\sigma^2}\} + \dfrac{1}{2} cos(\omega d^2)


        :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param omega: (``float``) Frequency of the oscillation.

        :param gamma: (``float``) Scale.

        :return: (``numpy.ndarray``) Kernel value/matrix between data points.
    """
    if sp.isspmatrix(x) and sp.isspmatrix(y):
        x = np.array(x.todense())
        y = np.array(y.todense())
    if not hasattr(x, "shape"):
        D = np.linalg.norm(x - y, ord=2)
    elif np.asarray(x).ndim == 0:
        D = np.abs(x - y)
    elif len(x.shape) >= 2 or len(y.shape) >= 2:
        D = cdist(x, y, metric="euclidean")
    else:
        D = np.linalg.norm(x - y, ord=2)
    return 0.5 * np.exp(-gamma * D**2) + 0.5 * np.cos(omega * D**2)


def exponential_absolute(x, y, sigma=2.0, gamma=None):
    """
    The exponential quadratic / radial basis kernel (RBF) kernel.

        .. math::
            k(\mathbf{x}, \mathbf{y}) = exp\{\dfrac{\|\mathbf{x} - \mathbf{y}\|^2}{\sigma^2} \}

        or

        .. math::
            k(\mathbf{x}, \mathbf{y}) = exp\{\gamma \|\mathbf{x} - \mathbf{y}\|^2 \}

        :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param sigma: (``float``) Length scale.

        :param gamma: (``float``) Scale.

        :return: (``numpy.ndarray``) Kernel value/matrix between data points.
    """
    if gamma is None:
        gamma = 1.0 / (2.0 * sigma ** 2)
    if sp.isspmatrix(x) and sp.isspmatrix(y):
        x = np.array(x.todense())
        y = np.array(y.todense())
    if not hasattr(x, "shape"):
        return np.exp(-gamma  * np.linalg.norm(x - y, ord=1))
    if np.asarray(x).ndim == 0:
        return np.exp(-gamma * np.absolute(x - y))
    if len(x.shape) >= 2 or len(y.shape) >= 2:
        return np.exp(-gamma * cdist(x, y, metric="cityblock"))
    return np.exp(-gamma * np.linalg.norm(x - y, ord=1))


rbf_kernel = exponential_kernel


def periodic_kernel(x, y, sigma=1, per=1, l=1):
    """
    The periodic kernel.
    Defined as in http://www.cs.toronto.edu/~duvenaud/cookbook/index.html.

        .. math::
            k(\mathbf{x}, \mathbf{y}) = \sigma^2 exp\{-2 \pi sin(\dfrac{\|\mathbf{x} - \mathbf{y}\|}{per})/l \}


        :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param per: (``float``) Period.

        :param l: (``float``) Length scale.

        :param sigma: (``float``) Variance.

        :return: (``numpy.ndarray``) Kernel value/matrix between data points.
    """
    if sp.isspmatrix(x) and sp.isspmatrix(y):
        x = np.array(x.todense())
        y = np.array(y.todense())
    if not hasattr(x, "shape"):
        return sigma**2 * np.exp(- 2 * np.sin(np.pi * np.absolute(x - y) / per) ** 2 / l ** 2)
    if np.asarray(x).ndim == 0:
        return sigma**2 * np.exp(- 2 * np.sin(np.pi * np.absolute(x - y) / per) ** 2 / l ** 2)
    if len(x.shape) >= 2 or len(y.shape) >= 2:
        return sigma ** 2 * np.exp(- 2 * np.sin(np.pi * cdist(x, y, metric="euclidean") / per) ** 2 / l ** 2)
    return sigma**2 * np.exp(- 2 * np.sin(np.pi * np.absolute(x - y) / per) ** 2 / l ** 2)


def matern_kernel(x, y, l=1.0, nu=1.5):
    """
    The Matern kernel wrapped from Scikit learn.

        .. math::
            k(\mathbf{x}, \mathbf{y}) = \sigma^2 \dfrac{2^{1-\nu}}{\Gamma{\nu}} (\sqrt{2\nu} \dfrac{d}{l})^{\nu} K_{\nu} (\sqrt{2\nu} \dfrac{d}{l})

        where {\Gamma } \Gamma is the gamma function, {K_{\nu }} K_{\nu }
        is the modified Bessel function of the second kind, and l and \nu are non-negative parameters of the covariance.

        :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param l: (``float``) Length scale.

        :param nu: (``float``) Differentiability of the kernel.

        :return: (``numpy.ndarray``) Kernel value/matrix between data points.
    """

    mk = Matern(length_scale=l, nu=nu)
    if sp.isspmatrix(x) and sp.isspmatrix(y):
        x = np.array(x.todense())
        y = np.array(y.todense())
    if not hasattr(x, "shape") or np.asarray(x).ndim == 0:
        x = np.reshape(np.array([x]), (1, 1))
    if not hasattr(y, "shape") or np.asarray(y).ndim == 0:
        y = np.reshape(np.array([y]), (1, 1))

    if np.asarray(x).ndim == 1: x = np.reshape(np.array([x]), (len(x), 1))
    if np.asarray(y).ndim == 1: y = np.reshape(np.array([y]), (len(y), 1))

    return mk(x, y)


def matern32_gpy(x, y, lengthscale=1):
    """
    Temp: GPy wrapper for the matern kernel.
    """
    x, y = correct_xy(x, y)
    k = GPy.kern.Matern32(input_dim=x.shape[1], lengthscale=lengthscale)
    return k.K(x, y)


def matern52_gpy(x, y, lengthscale=1):
    """
    Temp: GPy wrapper for the matern kernel.
    """
    x, y = correct_xy(x, y)
    k = GPy.kern.Matern52(input_dim=x.shape[1], lengthscale=lengthscale)
    return k.K(x, y)


def periodic_gpy(x, y, lengthscale=1, period=6.28):
    """
    Temp: GPy wrapper for the matern kernel.
    """
    x, y = correct_xy(x, y)
    k = GPy.kern.PeriodicExponential(input_dim=x.shape[1], lengthscale=lengthscale, period=period)
    return k.K(x, y)


def random_kernel(n):
    """
    Generate a random kernel matrix of shape ``(n, n)``.

    :param n: (``int``) Number of examples.

    :return: (``numpy.ndarray``) Random positive semidefinite kernel matrix of shape ``(n, n)``.
    """
    G = np.random.rand(n, n)
    return G.T.dot(G)


def center_kernel(K):
    """
    Center a kernel matrix.


    .. math::
        \mathbf{K}_{c} = (\mathbf{I}-\dfrac{\mathbf{11}^T}{n})\mathbf{K}(\mathbf{I}-\dfrac{\mathbf{11}^1}{n})
        

    :param K: (``numpy.ndarray``) Kernel matrix of shape ``(n, n)``.

    :return: (``numpy.ndarray``) Centered kernel for a sample of points.

    """
    m = int(K.shape[0])
    o = np.ones((m, 1))
    I = np.eye(m, m)
    Ic = (I-o.dot(o.T)/m)
    return Ic.dot(K).dot(Ic)


def center_kernel_low_rank(G):
    """
    Center a the feature matrix such that :math:`\mathbf{G}_c \mathbf{G}_c^T` is centered.

    .. math::
        \mathbf{G}_c = (\mathbf{I} - \dfrac{\mathbf{11}^T}{n})\mathbf{G}

    :param G: (``numpy.ndarray``) Low-rank approximation of the feature matrix of shape ``(n, k)``.

    :return: (``numpy.ndarray``) Centered low-rank approximation of the feature space.
    """
    return G - G.mean(axis=0)


def kernel_row_normalize(K):
    """
    Divide inner products of examples by their norm in the feature space,
    effectively computing angles. Applycable only to symmetric kernels.

    :param K: (``numpy.ndarray``) Kernel matrix of shape ``(n, n)``.

    :return: (``numpy.ndarray``) Row-normalized kernel for a sample of points.
    """
    assert K.shape[0] == K.shape[1]
    d = np.diag(K).reshape((K.shape[0], 1))
    Kn = np.sqrt(d.dot(d.T))
    return K / Kn


def kernel_to_distance(K):
    """
    Divide inner products of examples by their norm in the feature space,
    effectively computing angles. Applycable only to symmetric kernels.

    :param K: (``numpy.ndarray``) Kernel matrix or Kinterface of shape ``(n, n)``.

    :return: (``numpy.ndarray``) Distance matrix in the feature space induced by K.
    """
    assert K.shape[0] == K.shape[1]
    n = K.shape[0]
    d = K.diag() if hasattr(K, "diag") else np.diag(K)
    D = np.sqrt(-2 * K [:, :] + d.reshape((n, 1)) + d.reshape((1, n)))
    return D


def kernel_sum(x, y, kernels, kernels_args, kernels_weights=None):
    """
    Sum of arbitrary kernel functions.
    :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.
    :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.
    :param kernels: (``Iterable``) Iterable of pointers to kernels.
    :param kernels_args: (``Iterable``) Iterable with dictionaries, of the same length as `kernels`.
        Arguments are passed to kernels as kwargs.
    :param kernels_weights: (``Iterable``) Iterable with kernel weights, of the same length as `kernels`.
    :return:
    """
    assert len(kernels) == len(kernels_args)
    if kernels_weights is not None:
        return sum(w * k(x, y, **kw) for k, kw, w in zip(kernels, kernels_args, kernels_weights))
    else:
        return sum(k(x, y, **kw) for k, kw in zip(kernels, kernels_args))

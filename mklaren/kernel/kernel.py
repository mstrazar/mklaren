""" Methods related to calculation of kernel function values and kernel
    matrices.
"""

import numpy as np
import numpy.ma as ma
from itertools import product
import scipy.sparse as sp

def linear_kernel(x, y):
        """
        The linear kernel (the usual dot product in n-dimensional space).

        .. math::
            k(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T \mathbf{y}

        :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :return: (``numpy.ndarray``) Kernel value/matrix between data points.

        """
        if isinstance(x, int):
            return x * y
        if sp.isspmatrix(x):
            return np.array(x.dot(y.T).todense())
        else:
            return x.dot(y.T)


def poly_kernel(x, y, p=2, b=0):
        """
        The polynomial kernel.

        .. math::
            k(\mathbf{x}, \mathbf{y}) = (b + \mathbf{x}^T \mathbf{y})^p

        :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param p: (``float``) Polynomial degree.

        :param b: (``float``) Bias term.

        :return: (``numpy.ndarray``) Kernel value/matrix between data points.
        """
        if sp.isspmatrix(x):
            return np.array(x.dot(y.T).todense())**p
        if not hasattr(x, "shape"):
            return (x * y)**p
        else:
            return x.dot(y.T)**p


def sigmoid_kernel(x, y, b=1, c=0):
        """
        The sigmoid kernel.

        .. math::
            k(\mathbf{x}, \mathbf{y}) = tan(c \mathbf{x}^T \mathbf{y} + b)

        :param x: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param y: (``numpy.ndarray``) Data point(s) of shape ``(n_samples, n_features)`` or ``(n_features, )``.

        :param c: (``float``) Scale.

        :param b: (``float``) Bias term.

        :return: (``numpy.ndarray``) Kernel value/matrix between data points.
        """
        if sp.isspmatrix(x) and sp.isspmatrix(y):
            x = np.array(x.todense())
            y = np.array(y.todense())
        if not hasattr(x, "shape"):
            return np.tanh(b * x * y + c)
        else:
            return np.tanh(b * x.dot(y.T) + c)


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
        K = np.zeros((x.shape[0], y.shape[0]))
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                K[i, j] = np.exp(-gamma * np.linalg.norm(xi - yj, ord=2)**2)
        return K
    return np.exp(-gamma  * np.linalg.norm(x - y, ord=2)**2)

rbf_kernel = exponential_kernel



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
    m = float(K.shape[0])
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

import numpy as np
import numpy.ma as ma
from itertools import product
import scipy.sparse as sp

def linear_kernel(x1, x2):
        """
        :param x1:
            Data point(s) 1 shape(n_samples, n_features).
        :param x2:
            Data point(s) 2 shape(n_samples, n_features).
        :return:
            Linear kernel between data points shape(n_samples, n_samples).
        """
        if isinstance(x1, int):
            return x1 * x2
        if sp.isspmatrix(x1):
            return np.array(x1.dot(x2.T).todense())
        else:
            return x1.dot(x2.T)


def poly_kernel(x1, x2, p=2):
        """
        :param x1:
            Data point(s) 1 shape(n_samples, n_features).
        :param x2:
            Data point(s) 2 shape(n_samples, n_features).
        :param p
            Polynomial degree.
        :return:
            Polynomial kernel between data points shape(n_samples, n_samples).
        """
        if sp.isspmatrix(x1):
            return np.array(x1.dot(x2.T).todense())**p
        if not hasattr(x1, "shape"):
            return (x1 * x2)**p
        else:
            return x1.dot(x2.T)**p


def sigmoid_kernel(x1, x2, a=1, c=0):
        """
        :param x1:
            Data point(s) 1 shape(n_samples, n_features).
        :param x2:
            Data point(s) 2 shape(n_samples, n_features).
        :param a:
            Scale.
        :param c:
            Bias term.
        :return:
            Sigmoid kernel between data points shape(n_samples, n_samples).
        """
        if sp.isspmatrix(x1) and sp.isspmatrix(x2):
            x1 = np.array(x1.todense())
            x2 = np.array(x2.todense())
        if not hasattr(x1, "shape"):
            return np.tanh(a * x1 * x2 + c)
        else:
            return np.tanh(a * x1.dot(x2.T) + c)


def exponential_kernel(x1, x2, sigma=2.0, gamma=None):
    """

    Also known as RBF kernel.
        rbf_kernel

    :param x1:
        Data point(s) 1 shape(n_samples, n_features).
    :param x2:
        Data point(s) 2 shape(n_samples, n_features).
    :return:
        Linear kernel between data points shape(n_samples, n_samples).
    """

    if gamma is None:
        gamma = 1.0 / (2.0 * sigma ** 2)

    if sp.isspmatrix(x1) and sp.isspmatrix(x2):
        x1 = np.array(x1.todense())
        x2 = np.array(x2.todense())
    if not hasattr(x1, "shape"):
        return np.exp(-gamma  * np.linalg.norm(x1 - x2, ord=2)**2)
    if np.asarray(x1).ndim == 0:
        return np.exp(-gamma  * (x1 - x2)**2)
    if len(x1.shape) >= 2 or len(x2.shape) >= 2:
        K = np.zeros((x1.shape[0], x2.shape[0]))
        for i, x in enumerate(x1):
            for j, y in enumerate(x2):
                K[i, j] = np.exp(-gamma * np.linalg.norm(x - y, ord=2)**2)
        return K
    return np.exp(-gamma  * np.linalg.norm(x1 - x2, ord=2)**2)

rbf_kernel = exponential_kernel



def random_kernel(n):
    """
    :param n:
        Number of examples.
    :return:
        Random positive-semidefinit kernel matrix.
    """
    T = np.random.rand(n, n)
    return T.T.dot(T)


def center_kernel(K):
    """
    :param K:
        Kernel matrix.
    :return:
        Centered kernel for a sample of points.
    """
    m = float(K.shape[0])
    o = np.ones((m, 1))
    I = np.eye(m, m)
    Ic = (I-o.dot(o.T)/m)
    return Ic.dot(K).dot(Ic)


def center_kernel_low_rank(G):
    """
    :param G:
        Low-rank approximation of the feature space
    :return:
        Centered low-rank approximation of the feature space
    """
    return G - G.mean(axis=0)

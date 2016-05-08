"""
Linear algebra utility functions.
"""
from numpy import multiply, array, where, isinf, sum as npsum, max as npmax
from numpy import sqrt, random, eye, atleast_2d, zeros
from numpy.linalg import inv, norm
from itertools import product


def fro_prod(A, B):
    """
    The Frobenius product is an inner product between matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` of same shape.

    .. math::
        <\mathbf{A}, \mathbf{B}>_F = \sum_{i, j} \mathbf{A}_{ij} \mathbf{B}_{ij}

    :param A: (``numpy.ndarray``) a matrix.

    :param B: (``numpy.ndarray``) a matrix.

    :return: (``float``) Frobenius product value.
    """
    return npsum(multiply(A, B))


def fro_prod_low_rank(A, B):
    """
    The Frobenius product of kernel matrices induced by linear kernels on :math:`\mathbf{A}` and :math:`\mathbf{B}`:

    .. math::
        <\mathbf{AA}^T, \mathbf{BB}^T>_F

    Note :math:`A` and :math:`B` need not be of the same shape.

    :param A: (``numpy.ndarray``) a matrix.

    :param B: (``numpy.ndarray``) a matrix.

    :return: (``float``) Frobenius product value.
    """

    fp = 0
    for i in xrange(A.shape[1]):
        for j in xrange(B.shape[1]):
            gi = A[:, i]
            gj = B[:, j]
            fp += gi.dot(gj)**2
    return fp


def cosine_similarity(A, B):
    """
    Cosine similarity between matrices :math:`\mathbf{A}` and :math:`\mathbf{B}` defined in terms of the Frobenius product

    .. math::
        \dfrac{<\mathbf{A}, \mathbf{B}>_F} {<\mathbf{A}, \mathbf{A}>_F <\mathbf{B}, \mathbf{B}>_F}

    :param A: (``numpy.ndarray``) a matrix.

    :param B: (``numpy.ndarray``) a matrix.

    :return: (``float``) Cosine similarity value.
    """
    return fro_prod(A, B) / sqrt(fro_prod(A, A) * fro_prod(B, B))


def cosine_similarity_low_rank(a, b):
    """
    Cosine similarity between matrices from outer products of vectors :math:`\mathbf{a}` and :math:`\mathbf{b}`:

    .. math::
        \mathbf{A} = \mathbf{aa}^T

    .. math::
        \mathbf{B} = \mathbf{bb}^T

    .. math::
        \dfrac{<\mathbf{A}, \mathbf{B}>_F} {<\mathbf{A}, \mathbf{A}>_F <\mathbf{B}, \mathbf{B}>_F}

    :return: (``float``) Cosine similarity value.
    """
    enum = a.T.dot(b)**2
    denom = a.T.dot(a) * b.T.dot(b)
    return  1.0 * enum / denom


def cosine_similarity_low_rank_multi(G, y):
    """
    Cosine similarity between matrices from outer products of matrix :math:`\mathbf{G}` and vector :math:`\mathbf{y}`.

    :param G: (``numpy.ndarray``) Low-rank matrix.
    
    :param y: (``numpy.ndarray``) Column vector.
    
    :return: (``float``) Cosine similarity (kernel alignment).
    
    """
    enum = npsum(G.T.dot(y)**2)
    denom = y.T.dot(y) * sqrt(npsum([npsum(G[:, i] * G[:, j])**2
                                         for i, j in product(xrange(G.shape[1]), xrange(G.shape[1]))]))
    return 1.0 * enum / denom


def normalize(X):
    """
    Normalize :math:`\mathbf{X}` to have values between 0 and 1.

    :param X: (``numpy.ndarray``) Data matrix.

    :return: (``numpy.ndarray``) Normalize matrix.
    """
    return X / npmax(X)


def safe_divide(A, b):
    """
    Perform a safe float division between :math:`\mathbf{A}` and :math:`\mathbf{b}`.

    :param A: (``numpy.ndarray``) Scalar / Matrix.

    :param b: (``numpy.ndarray``) Scalar / Matrix.

    :return: (``float``) Matrix divided by b if b not zero else 0.
    """
    if isinstance(b, type(array([]))):
        D = 1.0 * multiply(A, b > 0)
        c = b.copy()
        c[where(b == 0)] = 1.0
        r =  multiply(D, 1.0 / c)
        r[where(isinf(r))] = 0
        return r
    return 1.0 * A / b if b**2 > 0 else 0 * A


def safe_func(x, f, val=0):
    """
    Evaluate a function defined on positive reals.

    :param x: (``float``) a value.

    :param f: (``callable``) Function to evaluate.

    :param val: (``float``) Return value when ``x`` is out of domain.

    :return: (``float``) Function value or out of domain.
    """

    if isinstance(x, type(array([]))):
        y = x.copy()
        y[where(y < 0)] = 0
        return f(y)
    elif x < 0:
        return val
    return f(x)


def outer_product(a, b):
    """
    Outer product between vectors :math:`\mathbf{a}` and :math:`\mathbf{b}`:

    :param a: (``numpy.ndarray``) column vector.

    :param b: (``numpy.ndarray``) column vector.

    :return: (``float``) Outer product of the appropriate dimension.
    """
    if len(a.shape) == 1 and len(b.shape) == 1:
        return a.reshape((len(a), 1)).dot(b.reshape((1, len(b))))
    else:
        return atleast_2d(a).dot(atleast_2d(b).T)


def woodbury_inverse(G, sigma2):
    """
    Matrix inversion using the Woodbury-Sherman-Morrison lemma.

    .. math::
        (\mathbf{GG}^T + \dfrac{1}{\sigma^2}  \mathbf{I})^{-1}

    Order of operations is important.

    :param G: (``numpy.ndarray``) Low-rank matrix.

    :param sigma2: (``float``) Noise / nugget term / sigma squared.

    :return: (``numpy.ndarray``) The solution to the above equation by inverting a shape ``(k, k)`` matrix.
    """
    assert sigma2 > 0
    n, k    = G.shape
    isigma2 = 1.0 / sigma2
    return isigma2 * eye(n, n) - isigma2 * G.dot(inv(sigma2 * eye(k, k) + G.T.dot(G)).dot(G.T))


def woodbury_inverse_full(Ai, U, Ci, V):
    """
    Matrix inversion lemma Woodbury-Sherman-Morrison in its general form.

    .. math::
        (\mathbf{A} + \mathbf{UCV})^{-1}

    :param Ai: (``numpy.ndarray``)  Inverse of a square matrix of shape ``(n, n)``.

    :param U: (``numpy.ndarray``)  Matrix of shape ``(n, k)``.

    :param Ci: (``numpy.ndarray``)  Inverse of square matrix of shape ``(k, k)``.

    :param V: (``numpy.ndarray``)  Matrix of shape ``(k, n)``.

    :return: (``numpy.ndarray``)  The solution to the above equation by inverting A and C. The solution is of shape ``(n, n)``.
    """
    return Ai - Ai.dot(U).dot(inv(Ci + V.dot(Ai).dot(U))).dot(V.dot(Ai))


def covariance_full_rank(K, sigma2):
    """
    Complete the kernel/covariance matrix to full rank by adding an indetity matrix.

    .. math::
        (\mathbf{K} + \sigma^2  \mathbf{I})^{-1}

    :param K: (``numpy.ndarray``)  Kernel matrix of shape ``(n, n)``.

    :param sigma2: (``float``) Added covariance.

    :return: (``numpy.ndarray``) Modified kernel matrix.
    """
    n = K.shape[0]
    return K + sigma2 * array(eye(n, n))


def ensure_symmetric(K):
    """
    Ensure a symmetric matrix. Small deviations in symmetry can present obstacles to ``numpy.svd`` used by samplers.

    .. math:
        \mathbf{K} = \dfrac{\math{K} + \math{K}^T}{2}

    :param K: (``numpy.ndarray``)  Kernel matrix of shape ``(n, n)``.

    :return: (``numpy.ndarray``)  Correction to C up to symmetry.
    """
    return (K + K.T) / 2.0


def qr(A):
    """
    Fast QR decomposition.

    :param A: (``numpy.ndarray``)  A matrix of shape ``(m, n)``.

    :return: (``tuple``) of (``numpy.ndarray``) orthonormal vectors (Q)  and (``numpy.ndarray``) upper triangular matrix (R).
    """
    m, n = A.shape
    Q = zeros((m, n))
    R = zeros((n, n))
    Q[:, 0] = safe_divide(A[:, 0], norm(A[:, 0]))
    R[0, 0] = norm(A[:, 0])
    for k in xrange(1, n):
        R[:k, k]  = Q[:, :k].T.dot(A[:, k])
        R[k, k]   = norm(A[:, k] - Q[:, :k].dot(R[:k, k]))
        Q[:, k]   = safe_divide(1.0, R[k, k]) * (A[:, k] -  Q[:, :k].dot(R[:k, k]))
    return Q, R

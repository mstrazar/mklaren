"""
    Linear algebra utility functions
"""
from numpy import multiply, array, where, isinf, sum as npsum, max as npmax
from numpy import sqrt, random, eye, atleast_2d, zeros
from numpy.linalg import inv, norm
from itertools import product


def fro_prod(X1, X2):
    """
    :param X1: array/matrix.
    :param X2: array/matrix.
    :return: Frobenius product between X1 and X2.
    """
    return npsum(multiply(X1, X2))


def fro_prod_low_rank(X1, X2):
    """
    :param X1: array/matrix.
    :param X2: array/matrix.
    :return: Frobenius product between X1 and X2, where X1 and X2 are low
             rank representations of square matrices X1 X1^T and X2 X2^T.
    """

    fp = 0
    for i in xrange(X1.shape[1]):
        for j in xrange(X2.shape[1]):
            gi = X1[:, i]
            gj = X2[:, j]
            fp += gi.dot(gj)**2
    return fp


def cosine_similarity(A, B):
    """
    Cosine similarity between matrix A and B.
    :param A: Matrix.
    :param B: Matrix.
    :return: Cosine similarity (kernel alignment).
    """
    return fro_prod(A, B) / sqrt(fro_prod(A, A) * fro_prod(B, B))

def cosine_similarity_low_rank(a, b):
    """
    Cosine similarity between matrices from outer products of vectors a, b.
    :param a: Column vector.
    :param b: Column vector.
    :return: Cosine similarity (kernel alignment).
    """
    enum = a.T.dot(b)**2
    denom = a.T.dot(a) * b.T.dot(b)
    return  1.0 * enum / denom

def cosine_similarity_low_rank_multi(G, y):
    """
    Cosine similarity between matrices from outer products of matrix G and vector y.
    :param G: Low-rank matrix.
    :param y: Column vector.
    :return: Cosine similarity (kernel alignment).
    """
    enum = npsum(G.T.dot(y)**2)
    denom = y.T.dot(y) * sqrt(npsum([npsum(G[:, i] * G[:, j])**2
                                         for i, j in product(xrange(G.shape[1]), xrange(G.shape[1]))]))
    return 1.0 * enum / denom


def normalize(X):
    """
    Normalize X to have values between 0 and 1.
    :param X: Data matrix.
    :return: Normalize matrix.
    """
    return X / npmax(X)


def safe_divide(A, b):
    """
    Perform a safe float division between a, b
    :param A: Scalar / Matrix
    :param b: Scalar / Matrix
    :return: Matrix divided by b if b not zero else 0
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
    :param x:
        Value.
    :param f:
        Function (callable).
    :param val
        Return value when x is out of domain.
    :return:
        Funciton value or out of domain.
    """

    if isinstance(x, type(array([]))):
        y = x.copy()
        y[where(y < 0)] = 0
        return f(y)
    elif x < 0:
        return val
    return f(x)


def outer_product(u, v):
    """
    Outer product between matrices/vectors u, v.
    :param u: Vector / Matrix
    :param v: Vector / Matrix
    :return: Outer product of the appropriate dimension.
    """
    if len(u.shape) == 1 and len(v.shape) == 1:
        return u.reshape((len(u), 1)).dot(v.reshape((1, len(v))))
    else:
        return atleast_2d(u).dot(atleast_2d(v).T)


def woodbury_inverse(G, sigma2):
    """
    Matrix inversion lemma Woodbury-Sherman-Morrison.

        inv((G * G.T)^2 + sigma**-1 * I)

    Order of operations is important.

    :param G: Rank-k approximation for symmetric matrix.
    :param sigma2: Prior variance (sigma squared).
    :return:
        The solution to the above equation by inverting k x k matrix.
    """
    assert sigma2 > 0
    n, k    = G.shape
    isigma2 = 1.0 / sigma2
    return isigma2 * eye(n, n) - isigma2 * G.dot(inv(sigma2 * eye(k, k) + G.T.dot(G)).dot(G.T))


def woodbury_inverse_full(Ai, U, Ci, V):
    """
    Matrix inversion lemma Woodbury-Sherman-Morrison in its general form.

        inv(A + UCV)

    :param Ai: Inverse of a square matrix n x n.
    :param U: Matrix n x k
    :param Ci: Inverse of square matrix k x k.
    :param V: Matrix k x n
    :return:
        The solution to the above equation by inverting A and C.
    """
    return Ai - Ai.dot(U).dot(inv(Ci + V.dot(Ai).dot(U))).dot(V.dot(Ai))


def covariance_full_rank(C, sigma2):
    """
    Complete covariance matrix to full rank by adding an indetity matrix.

    :param C:
        Symetric covariance matrix n x n.
    :param sigma2:
        Added covariance.
    :return:
        C + sigma2 * I.
    """
    n = C.shape[0]
    return C + sigma2 * array(eye(n, n))


def ensure_symmetric(C):
    """
    Ensure a symmetric matrix.
    Small deviations in symmetry can present obstacles to numpy.svd used
    by samplers.

    :param C:
    :return:
        Correction to C up to symmetry.
    """
    return (C + C.T) / 2.0


def qr(A):
    """
    Fast QR decomposition.
    :param A: m x n matrix
    :return:
        Q orthonormal vectors
        R upper triangular
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

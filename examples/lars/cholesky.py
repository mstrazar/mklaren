import numpy as np
from mklaren.util.la import safe_divide as div
from mklaren.mkl.mklaren import ssqrt
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import exponential_kernel


# TODO: this could be implemented more efficiently without iterating through inactive list.
def cholesky_steps(K, G, start, act, ina, order=None, no_steps=None):
    """
    Perform Cholesky steps for kernel K, starting from the existing matrix
    G at index k. Order of newly added pivots may be specified.

    Updates the matrix K in-place;

    :param K: Kernel matrix / interface.
    :param G: Existing Cholesky factors.
    :param start: Starting index.
    :param order: Possible to specify desired order. If not specified, standard gain criterion is taken.
    :param no_steps. Number of steps to take.
    :return: Updated Cholesky factors.
    """
    if order is None:
        no_steps = K.shape[0] if no_steps is None else no_steps
        have_order = False
    else:
        no_steps = len(order)
        have_order = True

    # Compute current diagonal
    d = K.diag() if isinstance(K, Kinterface) else np.diag(K).copy()
    D = d - np.sum(G*G, axis=1).ravel()

    for ki, k in enumerate(xrange(start, start + no_steps)):

        # Select best pivot
        i = order[ki] if have_order else ina[np.argmax(D[ina])]
        act.append(i)
        ina.remove(i)

        # Perform Cholesky step for the selected pivot
        j = list(ina)
        G[:, k] = 0
        G[i, k] = ssqrt(D[i])
        G[j, k] = div(1.0, G[i, k]) * (K[j, i] - G[j, :k].dot(G[i, :k].T))

        # Store previous and update diagonal
        D[j] = D[j] - (G[j, k] ** 2)
        D[i] = 0


def cholesky(K, rank=None):
    """ Wrap cholesky steps to perform a Cholesky decomposition of K."""
    n = K.shape[0]
    k = n if rank is None else rank
    G = np.zeros((n, k))
    cholesky_steps(K, G, start=0, act=[], ina=range(n), no_steps=k)
    return G


def test_cholesky():
    """ Simple test for complete Cholesky. """
    n = 10
    X = np.random.rand(n, 1)
    K = exponential_kernel(X, X, gamma=1)
    G = cholesky(K)
    norms = np.round(np.array([np.linalg.norm(K[:, :] - G[:, :i+1].dot(G[:, :i+1].T)) for i in range(n)]),
                     decimals=5)
    assert all(norms[:-1] >= norms[1:])
    assert np.linalg.norm(K-G.dot(G.T)) < 1e-5


def test_cholesky_interface():
    """ Simple test for complete Cholesky - Kinterface. """
    n = 10
    X = np.random.rand(n, 1)
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 1})
    G = cholesky(K)
    norms = np.round(np.array([np.linalg.norm(K[:, :] - G[:, :i + 1].dot(G[:, :i + 1].T)) for i in range(n)]),
                     decimals=5)
    assert all(norms[:-1] >= norms[1:])
    assert np.linalg.norm(K[:, :]-G.dot(G.T)) < 1e-5


def test_cholesky_steps_lookahead():
    """ Simple test for complete Cholesky - Kinterface in two steps.
        G, D, act, ina are updated in place. """
    n = 10
    X = np.random.rand(n, 1)
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 1})

    # Perform in-place Cholesky
    G = np.zeros((n, n))
    act = []

    cholesky_steps(K, G, start=0, act=act, ina=range(n), order=range(n/2))
    norms1 = np.round(np.array([np.linalg.norm(K[:, :] - G[:, :i + 1].dot(G[:, :i + 1].T)) for i in act]),
                      decimals=5)

    cholesky_steps(K, G, start=len(act), act=act, ina=range(n/2, n), order=range(n/2, n))
    norms2 = np.round(np.array([np.linalg.norm(K[:, :] - G[:, :i + 1].dot(G[:, :i + 1].T)) for i in range(n)]),
                      decimals=5)

    assert all(norms1[:-1] >= norms1[1:])
    assert all(norms2[:-1] >= norms2[1:])
    assert np.linalg.norm(K[:, :]-G.dot(G.T)) < 1e-5



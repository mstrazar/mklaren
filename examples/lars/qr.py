import numpy as np


def qr_steps(G, Q, R, max_steps=None, start=0):
    """
    Perform an in-place QR decomposition in steps.
    G at index k. Order of newly added pivots may be specified.

    Updates the matrix K in-place;

    :param G: Existing Matrix.
    :param Q: Existing array.
    :param R: Existing array.
    :param start: Starting index.
    :param max_steps. Number of steps to take. The order is implied by the columns in G, which match columns in Q.
    :return: Updated Cholesky factors.
    """
    max_steps = G.shape[1] if max_steps is None else max_steps
    for i in range(start, max_steps):
        if i == 0:
            Q[:, i] = G[:, i] / np.linalg.norm(G[:, i])
            R[i, i] = np.linalg.norm(G[:, i])
        else:
            R[:i, i] = Q[:, :i].T.dot(G[:, i])
            Q[:, i] = G[:, i] - sum([R[t, i] * Q[:, t] for t in range(i)])
            Q[:, i] /= np.linalg.norm(Q[:, i])
            R[i, i] = Q[:, i].T.dot(G[:, i])


def qr(G):
    """ QR decomposition of matrix G in steps. """
    n, k = G.shape
    Q = np.zeros((n, k))
    R = np.zeros((k, k))
    qr_steps(G, Q, R)
    return Q, R


def test_qr():
    n = 10
    k = 5
    G = np.random.rand(n, k)
    Q, R = qr(G)
    assert np.linalg.norm(G - Q.dot(R)) < 1e-5
    assert np.linalg.norm(Q.T.dot(Q) - np.eye(k)) < 1e-5


def test_qr_lookahead():
    n = 10
    k = 10
    G = np.random.rand(n, k)
    Q = np.zeros((n, k))
    R = np.zeros((k, k))

    k1 = 5
    qr_steps(G, Q, R, start=0, max_steps=k1)
    assert np.linalg.norm(Q[:, :k1].T.dot(Q[:, :k1]) - np.eye(k1)) < 1e-5
    assert np.linalg.norm(Q[:, k1:].T.dot(Q[:, k1:])) < 1e-5

    qr_steps(G, Q, R, start=k1, max_steps=None)
    assert np.linalg.norm(G - Q.dot(R)) < 1e-5
    assert np.linalg.norm(Q.T.dot(Q) - np.eye(k)) < 1e-5

import numpy as np
from scipy.linalg import solve_triangular


def solve_R(R):
    """ Invert up-to-permutation upper-triangular R. """
    k = R.shape[0]
    rs = np.sum(R != 0, axis=0).astype(int)-1
    P = np.argsort(rs)
    Ru = R[P, :][:, P]
    Rui = solve_triangular(Ru, np.identity(k), lower=False)
    return Rui[rs, :][:, rs]


def qr_reorder(Q, R, step, inxs):
    """ In-place reorder of first step columns of Q, R. R is in general no longer triangular.
        Must have max(inxs) < step. """
    Q[:, :step] = Q[:, :step][:, inxs]
    R[:step, :] = R[inxs, :]
    R[:, :step] = R[:, :step][:, inxs]
    return


def gqr_reorder(G, Q, R, step, inxs):
    """ In-place reorder of first step columns of G, Q, R to retain consistency.
        Must have max(inxs) < step. """
    G[:, :step] = G[:, :step][:, inxs]
    qr_reorder(Q, R, step, inxs)
    return


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
    max_steps = G.shape[1] - start if max_steps is None else max_steps
    i = start
    if i == 0:
        Q[:, i] = G[:, i] / np.linalg.norm(G[:, i])
        R[i, i] = np.linalg.norm(G[:, i])
        i += 1
    while i < start + max_steps:
        R[:i, i] = Q[:, :i].T.dot(G[:, i])
        Q[:, i] = G[:, i] - (R[:i, i] * Q[:, :i]).sum(axis=1)
        Q[:, i] /= np.linalg.norm(Q[:, i])
        R[i, i] = Q[:, i].T.dot(G[:, i])
        i += 1


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


def test_reorder():
    """ Reordering of columns in Chol/QR. """
    n = 10
    k = 5
    G = np.random.rand(n, k)
    Q, R = qr(G)
    assert np.linalg.norm(G - Q.dot(R)) < 1e-5

    # Reshuffle all
    inxs = np.random.choice(range(k), size=k, replace=False)
    G = G[:, inxs]
    Q = Q[:, inxs]
    R = R[inxs, :][:, inxs]
    assert np.linalg.norm(G - Q.dot(R)) < 1e-5

    # Reshuffle first few
    step = 3
    inxs = np.random.choice(range(step), size=step, replace=False)
    gqr_reorder(G, Q, R, step=step, inxs=inxs)
    assert np.linalg.norm(G - Q.dot(R)) < 1e-5


def test_solve_R():
    """ Reordering of columns in Chol/QR. """
    n = 10
    k = 5
    G = np.random.rand(n, k)
    Q, R = qr(G)
    step = 3
    inxs = np.random.choice(range(step), size=step, replace=False)
    gqr_reorder(G, Q, R, step=step, inxs=inxs)
    assert np.linalg.norm(G - Q.dot(R)) < 1e-5
    Ri = np.linalg.inv(R)
    Rui = solve_R(R)
    assert np.linalg.norm(Ri - Rui) < 1e-5

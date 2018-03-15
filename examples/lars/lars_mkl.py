import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from scipy.stats import multivariate_normal as mvn
from examples.lars.cholesky import cholesky_steps
from examples.lars.qr import qr_steps, qr_reorder, qr_orient
from warnings import warn
from mklaren.util.la import safe_divide as div
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import exponential_kernel

# TODO: add a custom gain function with penalty

hlp = """
    LARS with multiple kernels.
    Use a penalty function to influence the selection of new kernels.
"""


def f(p):
    """ Penalty function. """
    return p / (1.0 + p)


def mkl_qr(Ks, y, rank, delta, f=f):
    """ LARS-MKL fitting algorithm with a penalty function. """
    if delta == 1:
        raise ValueError("Unstable selection of delta. (delta = 1)")

    # Shared variables
    k = len(Ks)
    n = Ks[0].shape[0]
    Gs = []
    Acts = []
    Inas = []

    # Master QR decomposition and index of kernels (Permutation)
    Q = np.zeros((n, rank + k * delta))
    R = np.zeros((rank + k * delta, rank + k * delta))
    P = []

    # Current status and costs
    corr = np.zeros((k, ))
    penalty = np.zeros((k,))
    gain = np.zeros((k, n))

    # Look-ahead phase
    for j in range(len(Ks)):
        Gs.append(np.zeros((n, rank + delta)))
        Acts.append([])
        Inas.append(range(n))

        # Initial look-ahead setup ; one step per kernel
        if delta > 0:
            cholesky_steps(Ks[j], Gs[j], act=[], ina=range(n), max_steps=delta)

    # Iterations to fill active set
    for step in range(rank):
        for j in range(k):
            if delta > 0:
                G = Gs[j]
                L_delta = G.dot(G.T) - G[:, :step].dot(G[:, :step].T)
                IL = (np.eye(n) - Q[:, :step].dot(Q[:, :step].T)).dot(L_delta)
                B = np.round(np.linalg.norm(IL, axis=0) ** 2, 5)
                C = np.absolute(y.T.dot(IL)) ** 2
                gain[j, :] = div(C, B).ravel()
            else:
                gain[j, :] = Ks[j].diagonal() - (Gs[j] ** 2).sum(axis=1).ravel()

        # Select optimal kernel and pivot
        kern, pivot = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[kern, pivot] == 0:
            msg = "Iterations ended prematurely at step = %d < %d" % (step, rank)
            warn(msg)
            rank = step
            break
        assert len(Acts[kern]) == 0 or np.linalg.norm(gain[kern, Acts[kern]]) == 0
        assert pivot not in Acts[kern]

        # Select pivot and update Cholesky factors; try simplyfing
        G = Gs[kern]
        K = Ks[kern]
        P.append(kern)
        k_inx = np.where(np.array(P) == kern)[0]
        k_num = len(k_inx)
        k_start = k_num - 1

        # Update Cholesky
        G[:, k_start:] = 0
        cholesky_steps(K, G, start=k_start, act=Acts[kern], ina=Inas[kern], order=[pivot])
        qr_steps(G, Q, R, max_steps=1, start=k_start, qstart=step)
        assert np.linalg.norm(G[:, :k_num] - (Q.dot(R))[:, k_inx]) < 1e-5

        # Clear invalid columns and update lookahead steps;
        # Copy act/ina
        max_steps = min(delta, G.shape[1] - k_num)
        cholesky_steps(K, G,
                       act=list(Acts[kern]),
                       ina=list(set(range(n)) - set(Acts[kern])),
                       start=k_num,
                       max_steps=max_steps)
        assert np.linalg.norm(G[:, :k_num] - (Q.dot(R))[:, k_inx]) < 1e-5

        # Update current correlation
        corr[kern] = np.linalg.norm(Q[:, k_inx].T.dot(y)) ** 2

    # Correct lars order is based on including groups ; Gs are no longer valid
    # del Gs
    korder = np.argsort(-corr)
    porder = []
    for j in korder:
        porder.extend(list(np.where(np.array(P) == j)[0]))
    qr_reorder(Q, R, rank, porder)
    qr_orient(Q, R, y)
    assert rank == len(porder)

    # Return reduced approximatrion
    Q = Q[:, :rank]
    R = R[:rank, :rank]
    P = np.array(P)[porder]
    return Q, R, P


def mkl_lars(Q, P, y):
    """ Compute the group LARS path. """
    korder = list(set(P))
    pairs = zip(korder, korder[1:])
    path = np.zeros((len(korder) + 1, Q.shape[1]))
    t = np.sum(P == pairs[0][0])
    r = y.ravel()
    mu = 0

    # Compute steps
    for i, (k1, k2) in enumerate(pairs):
        c1 = np.linalg.norm(Q[:, P == k1].T.dot(r))
        c2 = np.linalg.norm(Q[:, P == k2].T.dot(r))
        alpha = 1 - (c2 / c1)
        path[i] = alpha * Q.T.dot(r).ravel()
        t += np.sum(P == k2)
        r = r - Q[:, :t].dot(path[i])
        mu = mu + Q[:, :t].dot(path[i])

    # Jump to least-squares solution
    path[-1] = Q.T.dot(r).ravel()
    r = r - Q.dot(path[-1])
    mu = mu + Q.dot(path[-1])
    assert np.linalg.norm(r) < 1e-3
    return path, mu


def mkl_lars_prediction():
    pass


# Unit tests
def test_lars_mkl():
    """ Simple test for LARS-kernel. """
    n = 100
    rank = 20
    delta = 5
    X = np.linspace(-10, 10, n).reshape((n, 1))
    Ks = [
        Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.6})[:, :], # short
        Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.1})[:, :], # long
        ]

    Kt = 1.0 + Ks[0] + 0.01 * Ks[1]
    y = mvn.rvs(mean=np.zeros(n,), cov=Kt).reshape((n, 1))

    plt.figure()
    plt.plot(y, ".")
    plt.show()

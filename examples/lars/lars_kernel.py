import numpy as np
from scipy.stats import multivariate_normal as mvn
from examples.lars.cholesky import *
from examples.lars.qr import *


# TODO: implement hard coded caluclations to check
# TODO: implement pivoting to cheaply update Chol/QR?
# TODO: do best elements always come from the lookahead set?

def lars_kernel(K, y, rank, delta):
    n = K.shape[0]
    G = np.zeros((n, rank + delta))
    Q = np.zeros((n, rank + delta))
    R = np.zeros((rank + delta, rank + delta))

    act = []
    ina = range(n)

    cholesky_steps(K, G, act=act, ina=ina, max_steps=rank + delta)
    # qr_steps(G, Q, R, max_steps=rank)
    qr_steps(G, Q, R, max_steps=rank + delta)
    r = y - Q[:, :rank].dot(Q[:, :rank].T.dot(y))
    print r.T.dot(Q)

    # Contribution has to be zero for active set.
    L_delta = G.dot(G.T) - G[:, :rank].dot(G[:, :rank].T)
    IL = (np.eye(n) - Q[:, :rank].dot(Q[:, :rank].T)).dot(L_delta)
    B = np.linalg.norm(IL, axis=0) ** 2
    C = np.linalg.norm(y.T.dot(IL), axis=0) ** 2
    gain = div(C, B)

    # Update active set w.r.t. maximal gain and increment decompositions
    print np.linalg.norm(K[:, :] - G.dot(G.T))
    print np.linalg.norm(G - Q.dot(R))
    print Q.T.dot(y)**2
    assert np.linalg.norm(gain[act][:rank]) == 0


def test_lars_kernel():
    n = 10
    rank = 5
    delta = 5
    X = np.linspace(-10, 10, n).reshape((n, 1))
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.3})
    y = mvn.rvs(mean=np.zeros(n,), cov=K[:, :]).reshape((n, 1))

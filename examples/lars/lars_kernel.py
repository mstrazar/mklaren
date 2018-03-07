import numpy as np
from scipy.stats import multivariate_normal as mvn
from examples.lars.cholesky import cholesky_steps
from examples.lars.qr import qr_steps, reorder_first
from examples.lars.lars_beta import plot_path, plot_residuals
from mklaren.util.la import safe_divide as div
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import exponential_kernel

np.set_printoptions(linewidth=2500)
np.set_printoptions(precision=2)

# TODO: implement hard coded caluclations to check
# TODO: do best elements always come from the lookahead set?
# TODO: implement pivoting to cheaply update Chol/QR?


def lars_kernel(K, y, rank, delta):
    """ Plain (suboptimal) implementation of LARS with kernels. """

    # Shared memory
    n = K.shape[0]
    G = np.zeros((n, rank + delta))
    Q = np.zeros((n, rank + delta))
    R = np.zeros((rank + delta, rank + delta))
    act = []
    ina = range(n)
    C_path = None

    # Initial look-ahead setup
    cholesky_steps(K, G, act=[], ina=range(n), max_steps=delta)
    qr_steps(G, Q, R, max_steps=delta)
    assert np.linalg.norm(G - Q.dot(R)) < 1e-5

    # Iterations to fill active set
    for step in range(rank):

        # Contribution has to be zero for active set
        L_delta = G.dot(G.T) - G[:, :step].dot(G[:, :step].T)
        IL = (np.eye(n) - Q[:, :step].dot(Q[:, :step].T)).dot(L_delta)
        B = np.round(np.linalg.norm(IL, axis=0) ** 2, 5)
        C = np.linalg.norm(y.T.dot(IL), axis=0) ** 2
        gain = div(C, B)
        jnew = np.argmax(gain)
        assert len(act) == 0 or np.linalg.norm(gain[act]) == 0
        assert jnew not in act

        # Select pivot and update
        G[:, step:] = 0
        Q[:, step:] = 0
        R[:, step:] = 0
        cholesky_steps(K, G, start=step, act=act, ina=ina, max_steps=1, order=[jnew])
        qr_steps(G, Q, R, max_steps=1, start=step)
        assert np.linalg.norm(G - Q.dot(R)) < 1e-5

        # Clear invalid columns and update lookahead steps;
        # Copy act/ina
        max_steps = min(delta, rank + delta - step)
        cholesky_steps(K, G,
                       act=list(act),
                       ina=list(set(range(n)) - set(act)),
                       start=step+1,
                       max_steps=max_steps)
        qr_steps(G, Q, R, max_steps=max_steps, start=step+1)
        assert np.linalg.norm(G - Q.dot(R)) < 1e-5

        # Apply correct LARS order - optimize with inserting into sorted list
        inxs = np.argsort(-np.absolute(Q[:, :step+1].T.dot(y).ravel()))
        act = list(np.array(act)[inxs[:step+1]])
        reorder_first(G, Q, R, step + 1, inxs)
        C_path = np.absolute(Q[:, :step+1].T.dot(y)).ravel()
        assert np.linalg.norm(G - Q.dot(R)) < 1e-5
        assert step == 0 or np.max(C_path[:-1] - C_path[1:]) >= 0

    # Compute regularization path
    sj = np.sign(Q[:, :rank].T.dot(y)).ravel()
    grad_path = C_path - np.concatenate((C_path[1:], np.array([0])))
    beta_path = np.zeros((rank, rank))
    for i in range(rank):
        beta_path[i, :i+1] = beta_path[i-1, :i+1] + grad_path[i] * sj[:i+1]
    mu = Q[:, :rank].dot(beta_path[-1].reshape((rank, 1)))
    return Q[:, :rank], R[:rank, :][:, :rank], beta_path, mu


# Plots
def plot_fit():
    """ Simple model fit. """
    n = 100
    rank = 20
    delta = 5
    X = np.linspace(-10, 10, n).reshape((n, 1))
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.3})[:, :]
    y = mvn.rvs(mean=np.zeros(n,), cov=K[:, :]).reshape((n, 1))
    Q, R, path, mu = lars_kernel(K, y, rank, delta)
    plot_path(path)
    plot_residuals(Q, y, path)


# Unit tests
def test_lars_kernel():
    """ Simple test for LARS-kernel. """
    n = 100
    rank = 20
    delta = 5
    X = np.linspace(-10, 10, n).reshape((n, 1))
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.3})[:, :]
    y = mvn.rvs(mean=np.zeros(n,), cov=K[:, :]).reshape((n, 1))
    Q, R, path, mu = lars_kernel(K, y, rank, delta)
    assert np.linalg.norm(Q.T.dot(y).ravel() - path[-1, :].ravel()) < 1e-5


def test_weigths_orthogonal():
    """ The weights in the orthogonal case change monotonically. """
    n = 100
    rank = 20
    delta = 5
    X = np.linspace(-10, 10, n).reshape((n, 1))
    K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.3})[:, :]
    y = mvn.rvs(mean=np.zeros(n,), cov=K[:, :]).reshape((n, 1))
    Q, R, path, mu = lars_kernel(K, y, rank, delta)
    for j in range(path.shape[1]):
        assert len(set(np.sign(path[:, j])) - {0}) <= 1

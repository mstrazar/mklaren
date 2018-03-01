import numpy as np
import scipy as sp
from examples.lars_vs_greedy.mklaren2 import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# class MklarenL1(MklarenNyst):
#
#     def __init__(self, rank, lbd=0, delta=10, debug=False):
#         MklarenNyst.__init__(self, rank, lbd, delta, debug)


# Simulate hyperparameters
rank = 30
delta = 5


def find_gradient(X, r, b):
    """ Find gradient over the bisector b and residual r. """
    c = X.T.dot(r).ravel()
    a = X.T.dot(b).ravel()
    C = max(c.ravel())
    A = max(a.ravel())
    t1 = (C - c) / (A - a)
    t2 = (C + c) / (A + a)
    valid1 = np.logical_and(t1 > 0, np.isfinite(t1))
    valid2 = np.logical_and(t2 > 0, np.isfinite(t2))
    grad = float("inf")
    if sum(valid1):
        grad = min(t1[valid1])
    if sum(valid2):
        grad = min(grad, min(t2[valid2]))
    return grad if np.isfinite(grad) else 0


def fit_mklaren(Ks, y):
    n = Ks[0].shape[0]

    # Set initial estimate and residual
    # self.sol_path = []
    # self.bias = y.mean()
    # regr = ones((n, 1)) * self.bias
    sol_path = []
    bias = y.mean()
    regr = ones((n, 1)) * bias
    residual = y - regr

    for t in range(min(rank, rank - delta)):

        # Expand full kernel matrices ; basis functions stored in rows ;
        # Simulate look-ahead of delta basis funtions
        # Xs = array([norm_matrix(Ka).T for K in Ks])
        Ka = K[:, :t + delta]
        Xs = array([norm_matrix(Ka).T])

        # Approximate basis functions
        Ga = sp.linalg.sqrtm(Ka.dot(np.linalg.inv(Ka[:t+delta, :])).dot(Ka.T))
        Xs_app = np.real(array([norm_matrix(Ga).T for K in Ks]))

        # Initial vector is selected by maximum *absolute* correlation
        Cs = Xs.dot(residual)

        # Make a deliberate bad selection in the beggining
        # rnd = np.random.randint(0, Cs.size)
        # q, i, _ = unravel_index(rnd, Cs.shape)

        # True best value from available
        q, i, _ = unravel_index(absolute(Cs).argmax(), Cs.shape)

        active = [(q, i)]
        Xa = hstack([sign(Xs[q, i, :].dot(residual)) * Xs[q, i, :].reshape((n, 1)) for q, i in active])

        # Compute bisector
        bisector, A = find_bisector(Xa)

        # Compute correlations with residual and bisector
        C = max(Xa.T.dot(residual))
        c = Xs_app.dot(residual)
        a = Xs_app.dot(bisector)
        assert C > 0

        # Select new basic function and define gradient
        # Negative values mean that the predictor must be turned for 180 degrees

        T1 = div((C + c), (A + a))
        T2 = div((C - c), (A - a))
        for q, i in active:
            T1[q, i] = T2[q, i] = float("inf")
        # T1[T1 <= 0] = float("inf")
        # T2[T2 <= 0] = float("inf")
        T = minimum(T1, T2)
        nq, ni, _ = unravel_index(T.argmin(), T.shape)
        grad = T[nq, ni]

        # Update state
        active = active + [(nq, ni)]
        regr = regr + grad * bisector
        residual = residual - grad * bisector

    # Plot current situation
    plt.figure()
    plt.plot(Xa, label="$x_a$")
    plt.plot(Xs[nq, ni, :], label="$x_{new}$")
    plt.plot(regr, label="$\mu$")
    plt.plot(residual, ".", label="$r$", color="black")
    plt.legend()




def test():
    N = 100
    n = 100
    noise = 0.03
    gamma = 0.1  # True bandwidth
    rank_range = [2, 3, 5, 8, 10, 20, 30]
    gamma_range = [gamma]

    # Generate data
    X = linspace(-10, 10, n).reshape((n, 1))
    K = exponential_kernel(X, X, gamma=gamma)
    w = randn(n, 1)
    f = K.dot(w) - K.dot(w).mean()
    noise_vec = randn(n, 1)  # Crucial: noise is normally distributed
    y = f + noise * noise_vec

    # Shuffle order and feed to model
    shuffle = np.random.choice(range(n), size=n, replace=False)
    K = K[shuffle, :][:, shuffle]
    y = y[shuffle]
    Ks = [K]


def example():
    """
    Example computation of late-coming vectors.
    :return:
    """
    y = array([2, 1, 3]).reshape((3, 1))
    X = eye(3)

    # Step 0
    b0, A = find_bisector(X[:, [0]])
    grad0 = find_gradient(X[:, [0, 1]], y, b0)
    mu0 = 0 + grad0 * b0
    r0 = y - mu0
    plot3(X, y, mu0, title="Step 0")

    # Step 1 -  WCS
    b1, A = find_bisector(X[:, [0, 1]])
    grad1 = find_gradient(X[:, [0, 1]], r0, b1)
    mu1 = mu0 + grad1 * b1
    r1 = y - mu1
    plot3(X, y, mu1, title="Step 1")

    # Step 2 - correction because X[:, 2] appears and is the best
    # Clearly, X[:, 2] must preceed previous predictors in the sequence, but the rest of the sequence
    # so far still stands - in the orthogonal case, even the order and SIZE of step updates do not change;
    d2, A = find_bisector(X)
    d2[:2] *= -1
    c = X.T.dot(r1)
    a = X.T.dot(d2)
    C = max(c)
    grad2 = (C-c[0]) / (A-a[0])
    mu2 = mu1 + grad2 * d2
    r2 = y - mu2
    plot3(X, y, mu2, title="Step 2a")

    # Step 3 - regular step accounting all the three variables
    b3, A = find_bisector(X)
    C = max(X.T.dot(r2))
    mu3 = mu2 + C/A * b3
    r3 = y - mu3
    plot3(X, y, mu3, title="Step 2b")


def plot3(X, r, mu, axislen=3, title="Step 0"):
    """ Plot data in 3D coordinate system. """

    # Coordinate system
    soa = np.array([[0, 0, 0] + list(X[:, 0]),
                    [0, 0, 0] + list(X[:, 1]),
                    [0, 0, 0] + list(X[:, 2])]) * axislen
    roa = np.array([[0, 0, 0] + list(r.ravel())])
    mua = np.array([[0, 0, 0] + list(mu.ravel())])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.quiver(*zip(*soa), color="black", alpha=0.2)
    ax.quiver(*zip(*roa), color="blue")
    ax.quiver(*zip(*mua), color="red")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')



def path3(X, mus, axislen=3, title="Solution path"):
    """ Plot data in 3D coordinate system. """

    # Coordinate system
    soa = np.array([[0, 0, 0] + list(X[:, 0]),
                    [0, 0, 0] + list(X[:, 1]),
                    [0, 0, 0] + list(X[:, 2])]) * axislen

    roa = np.array([[0, 0, 0] + list(r.ravel())])


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.quiver(*zip(*soa), color="black", alpha=0.2)
    ax.quiver(*zip(*roa), color="blue")
    ax.quiver(*zip(*mua), color="red")
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


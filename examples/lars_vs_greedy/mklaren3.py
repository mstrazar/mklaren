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


def find_gradient(X, r, b, act):
    """ Find gradient over the bisector b and residual r. """
    c = X.T.dot(r).ravel()
    a = X.T.dot(b).ravel()
    C = max(c[act].ravel())
    A = max(a[act].ravel())
    inxs = c > C
    if any(inxs):
        print("%s (%.3f)" % (str(c), C))
        print("Correlation condition violated with %.2f > %.2f!" % (max(c), C))
        raise ValueError
    t1 = (C - c) / (A - a)
    t2 = (C + c) / (A + a)
    valid1 = np.logical_and(t1 > 0, np.isfinite(t1))
    valid2 = np.logical_and(t2 > 0, np.isfinite(t2))
    grad = float("inf")
    if sum(valid1):
        grad = min(t1[valid1])
    if sum(valid2):
        grad = min(grad, min(t2[valid2]))
    if not np.isfinite(grad):
        raise Exception("Infinite gradient!")
    return grad, C


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


def orthog_lars_simple(X, y):
    """
    Simple orthogonal LARS with full information. Solves the path in one sorting step.
    Bisector is a simple sum of orthogonal components.
    X is an orthogonal matrix.
    :return:
    """
    n = X.shape[0]
    X = X * np.sign(X.T.dot(y)).ravel()
    c_all = X.T.dot(y).ravel()
    act = np.argsort(-c_all)
    grads = c_all[act] - np.concatenate((c_all[act][1:], np.array([0])))
    path = np.zeros((n, n))
    for i, a in enumerate(act):
        bisec = X[:, act[:i+1]].sum(axis=1)
        p = path[i-1] if i > 0 else 0
        path[i] = p + grads[i] * bisec.ravel()
    return path


def orthog_lars_sequential():
    """
    Example computation of late-coming vectors.
    Y determines the order in which variables enter the model.
    Have to remember what the value of C was when each variable entered the model.
    Only makes sense if there is some smart selection of the variables based on the current estimate.
    """
    y = array([4, 2, 3, 1]).reshape((4, 1))
    n = len(y)
    X = eye(n)
    r = y.reshape((n, 1))
    mu = np.zeros((n, 1))

    # Path variables; to be reshuffled accordingly;
    grad_path = np.zeros((n,))
    C_path = np.zeros((n,))
    act = [0]
    for step in range(n):
        print("Step: %d" % step)
        nxt = [step + 1]
        c_act = X[:, act].T.dot(y).ravel()
        c_nxt = X[:, nxt].T.dot(y).ravel()
        C_a = max(c_act)
        C_n = max(c_nxt)
        if C_a > C_n:
            grad = C_a - C_n
            mu[act] += grad
            grad_path[step] = grad
            C_path[step] = C_a
            r = y - mu
        else:
            # Update residual too, in order to evaluate for new columns?
            inxs = C_path > C_n      # True: variables in correct positions
            p = np.argmin(inxs)      # Position of the current variable
            grad_01 = C_path[p - 1] - C_n if p > 0 else None
            grad_12 = C_n - C_path[p]  # Positive by construction
            if p > 0:
                C_path = np.hstack([C_path[:p],
                                    np.array([C_n]),
                                    C_path[p+1:]])
                grad_path = np.hstack([grad_path[:p],
                                       np.array([grad_01]),
                                       np.array([grad_12]),
                                       grad_path[p + 1:]])
                act = act[:p] + [step + 1] + act[p:]
                # etc.


def orthog_lars_simple_test():
    n = 100
    y = np.sort(np.random.rand(n))
    X, _, _ = np.linalg.svd(np.random.rand(n, n))
    paths = orthog_lars_simple(X, y)
    norms = paths.sum(axis=1)

    # Plot regularization path
    plt.figure()
    plt.plot(norms / norms.max())
    plt.xlabel("Step")
    plt.ylabel("$\|f\|_1$ / $\|f_{LS}\|_1$")
    plt.grid()

    # Plot function approximation
    plt.figure()
    plt.plot(y, ".", color="black")
    for pi in range(0, n, 10):
        plt.plot(paths[pi], "-", color="blue", alpha=0.2)
    plt.xlabel("Index")
    plt.ylabel("y")


def qr_order():
    """ Order or columns in the QR decomposition can have an effect on the solution in the original space. """
    X = np.array(([1, 1], [1, 0]))
    # Q = np.eye(2)       # "Wrong" QR step
    Q = 1.0/np.sqrt(2) * np.array(([1, 1], [1, -1])) # "Right" QR step
    R = Q.T.dot(X)
    y = np.array([0.7, 0.8]).reshape((2, 1))

    # LAR solution path in the Q space
    # W = np.array([[0, 0], [0, 0.1], [0.7, 0.8]]).T      # "Wrong" QR step
    W = np.array([[0, 0], [0.9, 0.0], [1.06, -0.07]]).T # "Right" QR step

    # Solution path in the original space
    Wo = np.linalg.inv(R).dot(W)

    for Ax, Sw in [(X, Wo)]:
        plt.figure()
        plt.plot([0, Ax[0, 0]], [0, Ax[1, 0]], "k-")
        plt.plot([0, Ax[0, 1]], [0, Ax[1, 1]], "k-")
        a = np.array([0, 0])
        for w in Sw.T:
            x = Ax.dot(w)[::-1]
            plt.plot([a[0], (x)[0]], [a[1], (x)[1]], "r.-")
            a = x
        plt.xlim(-0.5, 1.5)
        plt.ylim(-0.5, 1.5)
        plt.grid()
    plt.show()




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

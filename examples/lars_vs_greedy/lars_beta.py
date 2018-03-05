import numpy as np
import matplotlib.pyplot as plt


def find_bisector(X):
    """ Find bisector and the normalizing constant for vectors in X."""
    # Compute bisector
    Ga = X.T.dot(X)
    Gai = np.linalg.inv(Ga)
    A = 1.0 / np.sqrt(Gai.sum())
    omega = A * Gai.dot(np.ones((X.shape[1], 1)))
    bisector = X.dot(omega)
    return bisector, A


def find_gradient(X, r, b, act):
    """ Find gradient over the bisector b and residual r. """
    c = X.T.dot(r).ravel()
    a = X.T.dot(b).ravel()
    C = max(np.absolute(c[act].ravel()))
    A = max(np.absolute(a[act].ravel()))
    ina = list(set(range(X.shape[1])) - set(act))
    if A <= 0:
        raise ValueError("A = %f" % A)
    inxs = c > C
    if any(inxs):
        print("%s (%.3f)" % (str(c), C))
        print("Correlation condition violated with %.2f > %.2f!" % (max(c), C))
        raise ValueError
    t1 = ((C - c) / (A - a))[ina]
    t2 = ((C + c) / (A + a))[ina]
    valid1 = np.logical_and(t1 > 0, np.isfinite(t1))
    valid2 = np.logical_and(t2 > 0, np.isfinite(t2))
    grad = float("inf")
    if sum(valid1):
        grad = min(t1[valid1])
    if sum(valid2):
        grad = min(grad, min(t2[valid2]))
    return grad


def lars_beta(X, y):
    """
    General LARS with full information.
    X is a matrix with positive correlation to y.
    :return: Solution path of implicit regression weigths.
    """
    n = X.shape[0]
    r = y.reshape((n, 1))
    mu = np.zeros((n, 1))
    act = [np.argmax(np.absolute(X.T.dot(r)))]
    ina = list(set(range(n)) - set(act))
    path = np.zeros((n, n))

    for step in range(n):
        sj = np.sign(X[:, act].T.dot(r)).T.ravel()
        Xa = X[:, act] * sj
        b, A = find_bisector(Xa)
        if step == n - 1:
            C_a = np.max(np.absolute(X.T.dot(r)))
            grad = C_a / A
        else:
            grad = find_gradient(X, r, b, act)
            rnew = r - grad * b
            j = ina[int(np.argmax(np.absolute(X[:, ina].T.dot(rnew))))]
            act.append(j)
            ina.remove(j)
        mu = mu + grad * b
        r = r - grad * b
        path[step, :] = np.linalg.lstsq(X, mu)[0].ravel()
    return np.round(path, 3), mu


# Comparisons
def compare_original_qr():
    """ Test the LAR algorithm in the orthogonal and general case. """
    n = 100
    X = np.random.rand(n, n)
    X = X / np.linalg.norm(X, axis=0).ravel()
    y = np.random.rand(n, 1)
    y = np.sort(y - y.mean(), axis=0)
    Q, R = np.linalg.qr(X)
    path_orig, mu_orig = lars_beta(X, y)
    path_q, mu_q = lars_beta(Q, y)
    path_q_mapped = (np.linalg.inv(R).dot(path_q.T)).T

    plot_path(path_orig, tit="X")
    plot_path(path_q, tit="Q")
    plot_path(path_q_mapped, tit="X-Q mapped")
    plot_residuals(X, y, path_orig, tit="X")
    plot_residuals(Q, y, path_q, tit="Q")
    plot_residuals(X, y, path_q_mapped, tit="X-Q mapped")


def compare_nonrandom():
    """ Compare the detection of a ground truth model. """
    n = 100
    X = np.random.rand(n, n)
    X = X / np.linalg.norm(X, axis=0).ravel()

    y = X[:, n/2].reshape((n, 1)) + np.random.rand(n, 1) * 0.1
    path_orig, mu_orig = lars_beta(X, y)

    Q, R = np.linalg.qr(X)
    path_q, mu_q = lars_beta(Q, y)
    path_q_mapped = (np.linalg.inv(R).dot(path_q.T)).T

    plot_path(path_orig, tit="X")
    plot_path(path_q, tit="Q")
    plot_path(path_q_mapped, tit="X-Q mapped")

    plot_residuals(X, y, path_orig, tit="X")
    plot_residuals(X, y, path_q_mapped, tit="X-Q mapped")


def compare_correlated():
    from mklaren.kernel.kernel import exponential_kernel
    from scipy.stats import multivariate_normal as mvn
    n = 30
    noise = 0.01
    gamma = 0.1
    X = np.linspace(-10, 10, n).reshape((n, 1))
    K = exponential_kernel(X, X, gamma=gamma)
    y = mvn.rvs(mean=np.zeros(n, ), cov=K + noise * np.eye(n))
    Q, R = np.linalg.qr(K)
    print "cond(K):", np.linalg.cond(K)
    print "cond(Q):", np.linalg.cond(Q)

    # Solve problem
    path_orig, mu_orig = lars_beta(K, y)
    path_q, mu_q = lars_beta(Q, y)
    path_q_mapped = (np.linalg.inv(R).dot(path_q.T)).T

    # Kernels can be highly correlated objects
    C = np.corrcoef(K)
    plt.hist(C.ravel())

    plot_path(path_orig, tit="K")
    plot_path(path_q, tit="Q")
    plot_path(path_q_mapped, tit="K-Q mapped")

    plot_residuals(K, y, path_orig, tit="K")
    plot_residuals(K, y, path_q_mapped, tit="K-Q mapped")
    plot_residuals(Q, y, path_q, tit="Q")

    plt.figure()
    plt.plot(y, "-")
    plt.plot(mu_q, ".")
    plt.xlabel("x")
    plt.ylabel("f(x)")



# Plots
def plot_path(path, tit=""):
    """ Plot weigths as solution paths."""
    plt.figure()
    plt.title(tit)
    P = path.T
    for p in P:
        plt.plot(p, ".-")
    plt.ylim((np.min(P), np.max(P)))
    plt.xlabel("Model capacity $\\rightarrow$")
    plt.ylabel("Feature size")
    plt.grid()

    plt.figure()
    plt.title(tit)
    norms = [np.linalg.norm(p, ord=1) for p in path]
    plt.plot(norms, ".-")
    plt.xlabel("Model capacity $\\rightarrow$")
    plt.ylabel("$\|\\beta\|_1$")
    plt.grid()


def plot_residuals(X, y, path, tit=""):
    """ Plot model residuals given the path."""
    mus = X.dot(path.T).T
    norms = [np.linalg.norm(p, ord=1) for p in path]
    res = [np.linalg.norm(y.ravel() - mu.ravel()) for mu in mus]
    plt.figure()
    plt.title(tit)
    plt.plot(res, ".-")
    plt.xlabel("Model capacity $\\rightarrow$")
    plt.ylabel("$\|h(x)-y\|_2$")
    plt.grid()

    plt.figure()
    plt.title(tit)
    plt.plot(norms, res, ".-")
    plt.xlabel("$\|\\beta\|_1$")
    plt.ylabel("$\|h(x)-y\|_2$")
    plt.grid()


# Unit tests
def test_lars_beta_full():
    n = 5
    X = np.random.rand(n, n)
    X = X / np.linalg.norm(X, axis=0).ravel()
    y = np.random.rand(n, 1)
    y = np.sort(y - y.mean(), axis=0)
    path, mu = lars_beta(X, y)
    assert np.linalg.norm(mu - y) < 1e-3


def test_bisector():
    n = 5
    X = np.random.rand(n, n)
    for inxs in [range(i) for i in range(1, n)]:
        b, A = find_bisector(X[:, inxs])
        assert np.var(X[:, inxs].T.dot(b)) < 1e-5
        assert (np.linalg.norm(b) - 1) < 1e-5


def test_find_gradient():
    n = 5
    y = np.random.randn(n, 1)
    X, _, _ = np.linalg.svd(np.random.randn(n, n))
    r = y - y.mean()
    X = X * np.sign(X.T.dot(r)).T.ravel()
    c = np.absolute(X.T.dot(r))
    act = list(np.where(c == np.max(c))[0])
    ina = list(set(range(n)) - set(act))
    b, A = find_bisector(X[:, act])
    grad = find_gradient(X, r, b, act)
    rnew = r - grad * b
    cnew = X.T.dot(rnew)
    assert np.max(np.absolute(cnew[act])) - np.max(np.absolute(cnew[ina])) < 1e-3


def test_all():
    for i in range(1000):
        test_bisector()
        test_find_gradient()
        test_lars_beta_full()

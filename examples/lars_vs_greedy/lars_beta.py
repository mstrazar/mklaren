import numpy as np


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


def lars_beta_sequential(X, y):
    """
    General LARS with sequential information.
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
            j = ina[np.argmax(np.absolute(X[:, ina].T.dot(rnew)))]
            act.append(j)
            ina.remove(j)
        mu = mu + grad * b
        r = r - grad * b
        path[step, :] = np.linalg.lstsq(X, mu)[0].ravel()
    return np.round(path, 3), mu


### Unit tests

def test_lars_beta_sequential():
    n = 5
    X = np.random.rand(n, n)
    X = X / np.linalg.norm(X, axis=0)
    y = np.random.rand(n, 1)
    y = np.sort(y - y.mean(), axis=0)
    path, mu = lars_beta_sequential(X, y)
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
        test_lars_beta_sequential()

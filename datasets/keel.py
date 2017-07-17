import gzip
from glob import glob
from numpy import genfromtxt
from os.path import join, realpath, dirname, basename

KEEL_PATH = join(dirname(realpath(__file__)), "keel")
KEEL_DATASETS = map(lambda p: basename(p), glob(join(KEEL_PATH, "*")))

def load_keel(name="abalone", n=None):
    """
    Load an keel dataset.
    :param name: Dataset name.
    :param n: Maximum number of examples.
    :return: Dataset in standard form.
    """
    fp = gzip.open(join(KEEL_PATH, name, "%s.dat.gz" % name))
    data = genfromtxt(fp, delimiter=",", skip_header=1, dtype=float, comments="@")
    X = data[:, :-1]
    y = data[:, -1].ravel()
    labels = map(lambda i: "a%d" % i, range(X.shape[1]))

    if n is not None and n < X.shape[0]:
        X = X[:n, :]
        y = y[:n]

    return {
        "data": X, "target": y, "labels": labels
    }
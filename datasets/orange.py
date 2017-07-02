from numpy import genfromtxt
from os.path import join, realpath, dirname

ORANGE_PATH = join(dirname(realpath(__file__)), "orange")

def load_ionosphere(n=None):
    """
    Load the ionosphere dataset.
    :param n: Maximum number of examples.
    :return: Dataset in standard form.
    """
    header = genfromtxt(join(ORANGE_PATH, "ionosphere.csv"),
                      delimiter=",", skip_header=0, dtype=str, max_rows=1)
    data = genfromtxt(join(ORANGE_PATH, "ionosphere.csv"),
                      delimiter=",", skip_header=1, dtype=float)
    X = data[:, :-1]
    y = data[:, -1].ravel()
    if n is not None and n < X.shape[0]:
        X = X[:n, :]
        y = y[:n]
    labels = header[:-1]
    assert len(labels) == X.shape[1]
    return {
        "data": X, "target": y, "labels": labels
    }
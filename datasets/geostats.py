hlp = """
    Source: http://www.leg.ufpr.br/doku.php/pessoais:paulojus:mbgbook:datasets
"""

from numpy import genfromtxt
from os.path import join, realpath, dirname
import scipy.stats as st

GEO_PATH = join(dirname(realpath(__file__)), "geostats")

def load_geostats(name="camg", n=None):
    """
    Load the camg dataset.
    :param name: Dataset name.
    :param n: Maximum number of examples.
    :return: Dataset in standard form.
    """
    fp = join(GEO_PATH, name, "%s.csv" % name)
    data = genfromtxt(open(fp), delimiter=",", skip_header=1, dtype=float)
    X = data[:, :2]
    y = data[:, 2].ravel()
    labels = map(lambda i: "a%d" % i, range(X.shape[1]))

    # Jitter this data
    if n is not None and n < X.shape[0]:
        X = X[:n, :]
        y = y[:n]
    return {"data": X, "target": y, "labels": labels}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = load_geostats(name="sic467", n=None)
    X = data["data"]
    y = st.zscore(data["target"])

    plt.figure()
    for i in range(X.shape[0]):
        plt.plot(X[i, 0], X[i, 1],  "k.", alpha=0.2, markersize=15*y[i],)
    plt.show()


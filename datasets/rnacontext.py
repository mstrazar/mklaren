import gzip
from glob import glob
from numpy import genfromtxt
from os.path import join, realpath, dirname, basename

RNA_PATH = join(dirname(realpath(__file__)), "rnacontext", "full")
RNA_DATASETS = map(lambda p: basename(p), glob(join(RNA_PATH, "*")))

def load_rna(name, n=None):
    """
    Load an keel dataset.
    :param name: RNA context Dataset name.
    :param n: Maximum number of examples.
    :return:
        Dataset in standard form.
        X is an array of strings.
    """
    fp = gzip.open(join(RNA_PATH, name))
    y = genfromtxt(fp, delimiter="\t", dtype=float, comments="@", usecols=[0])
    fp = gzip.open(join(RNA_PATH, name))
    X = genfromtxt(fp, delimiter="\t", dtype=str, comments="@", usecols=[1])
    if n is not None and n < X.shape[0]:
        X = X[:n, :]
        y = y[:n]
    return {
        "data": X, "target": y, "labels": ["sequence"],
    }
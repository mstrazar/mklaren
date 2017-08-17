import gzip
from glob import glob
from numpy import genfromtxt
from os.path import join, realpath, dirname, basename

# RNA_PATH = join(dirname(realpath(__file__)), "rnacontext", "full")
RNA_PATH = join(dirname(realpath(__file__)), "rnacontext", "weak")
RNA_DATASETS = map(lambda p: basename(p), glob(join(RNA_PATH, "*")))

# Optimal K as listed in RNAcontext article
RNA_OPTIMAL_K = {
    'VTS1_data_full_AB.txt.gz': 7,
    'SLM2_data_full_AB.txt.gz': 8,
    'RBM4_data_full_AB.txt.gz': 7,
    'SF2_data_full_AB.txt.gz': 5,
    'Fusip_data_full_AB.txt.gz': 10,
    'HuR_data_full_AB.txt.gz': 9,
    'PTB_data_full_AB.txt.gz': 5,
    'U1A_data_full_AB.txt.gz': None,
    'YB1_data_full_AB.txt.gz': None,
}

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
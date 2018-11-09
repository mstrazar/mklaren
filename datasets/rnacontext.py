import gzip
import pickle
import itertools as it
import scipy.sparse as sp
from glob import glob
from numpy import genfromtxt, int16
from os.path import join, realpath, dirname, basename

# RNA_PATH = join(dirname(realpath(__file__)), "rnacontext", "full")
RNA_PATH = join(dirname(realpath(__file__)), "rnacontext", "weak")
RNA_DATASETS = map(lambda p: basename(p), glob(join(RNA_PATH, "*.txt.gz")))
RNA_ALPHABET = {"A": 0, "C": 1, "G": 2, "T": 3}

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
        X = X[:n]
        y = y[:n]
    return {
        "data": X, "target": y, "labels": ["sequence"],
    }


def kmer_index(K):
    """
    K-mer (substring) to index in a sparse matrix of spectrum kernel feature space.
    :param K: K-mer length (substring).
    :return: Index.
    """
    kmer2index = lambda kmer: sum((RNA_ALPHABET[k] * 4**i for i, k in enumerate(kmer[::-1])))
    kmers = map(lambda t: "".join(t), it.product(*(K*[RNA_ALPHABET.keys()])))
    return dict(map(lambda km: (km, kmer2index(km)), kmers))


def convert_sparse(X, K, f_out=None):
    """
    Convert a string array into a scipy sparse matrix.
    :param X: String array.
    :param K: K-mer (substring) length.
    :param f_out: Output file.
    :return:
    """
    seq2kmers = lambda seq: map(lambda t: "".join(t), zip(*[seq[i:] for i in range(K)]))
    index = kmer_index(K)
    Y = sp.dok_matrix((len(X), len(index)), dtype=int16)
    for i, seq in enumerate(X):
        cols = map(lambda km: index[km], seq2kmers(seq))
        for j in cols: Y[i, j] += 1
    Y = Y.tocsr()
    if f_out is None:
        return Y
    else:
        fp = gzip.open(f_out, "w") if ".gz" in f_out else open(f_out, "w")
        pickle.dump(Y, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()
        print("Written %s" % f_out)
        return

def dataset2spectrum(dset, K):
    """
    Convert dataset name to spectrum location
    :param dset: dataset original name
    :param K: K-mer (substring) length
    :return:
    """
    return join(RNA_PATH, "spectrum", basename(dset).split(".")[0] + ".%d" % K + ".pkl.gz")


def convert_datasets(kmin=1, kmax=10):
    """
    Convert a dataset to spectrum.
    :param d_out: Output directory.
    :param kmin: minimal K.
    :param kmax: maximal K.
    :return:
    """
    for dset in RNA_DATASETS:
        X = load_rna(dset)["data"]
        for K in range(kmin, kmax+1):
            f_out = dataset2spectrum(dset, K)
            convert_sparse(X, K, f_out=f_out)


if __name__ == "__main__":
    print("Converting datasets to spectrum ...")
    convert_datasets()
from numpy import zeros, array, argsort
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse.linalg import norm as spnorm
from os import environ
from os.path import join, dirname, realpath
from hashlib import md5
from gzip import open as gzopen


BLITZER_PATH = join(dirname(realpath(__file__)), "blitzer")


def md5hash(s):
    """
    Digest strings for unique feature selection.

    :param s:
        String to digest
    :return:
        MD5 digested string.
    """
    m = md5()
    m.update(s)
    return m.digest()

def load_keys(dataset="books"):
    """
    Read all possible keys from files.

    :param dataset:
        Dataset name, from
            books
            dvd
            electronics
            kitchen
    :return:
        Dictionary of keys and indices.
        Dictionary of file lengths.
    """
    keysd = dict()
    lengthd = dict()
    labelsd = dict()

    wordcount = 0


    for split in ["train", "test"]:
        fp        = gzopen(join(BLITZER_PATH, dataset, split))
        linecount = 0
        line      = fp.readline()
        while line:
            for token in line.strip().split(" "):
                ky, _ = token.split(":")
                if (ky != "#label#") and (ky not in keysd):
                    keysd[ky] = wordcount
                    labelsd[wordcount] = ky
                    wordcount = wordcount + 1
            line = fp.readline()
            linecount += 1

        lengthd[split] = linecount
    return keysd, labelsd, lengthd


def load_blitzer(dataset="books", n=None, max_features=None, tol=1e-5):
    """
    Load the Blitzer sentiment analysis dataset.
    All matrices are scipy.sparse.

    :param dataset:
        Dataset name.
    :param n:
        Read first n rows.
    :param tol
        Tolerance with equal columns in the training set.
    :param max_features:
        Read max_features most abundant features.
    :return:
        Dictionary {"data": X, "target": y,
                    "data_test": X_test, "target": y_test}


    """
    path = join(BLITZER_PATH, dataset)
    keysd, labelsd, lengthd = load_keys(dataset)

    p = len(keysd)
    n_train = lengthd["train"]
    n_test = lengthd["test"]

    if n is None:
        X_train = dok_matrix((n_train, p), dtype=int)
        y_train = zeros((n_train, ), dtype=int)

        X_test = dok_matrix((n_test, p), dtype=int)
        y_test = zeros((n_test, ), dtype=int)
    else:
        X_train = dok_matrix((n, p), dtype=int)
        X_test = dok_matrix((n, p), dtype=int)
        y_train = zeros((n, ), dtype=int)
        y_test = zeros((n, ), dtype=int)

    for split, X, y in zip(["train", "test"], [X_train, X_test], [y_train, y_test]):
        fp        = gzopen(join(BLITZER_PATH, dataset, split))
        linecount = 0
        line      = fp.readline()
        while line:
            for token in line.strip().split(" "):
                ky, val = token.split(":")
                val = int(float(val))
                if ky == "#label#":
                    y[linecount] = val
                else:
                    X[linecount, keysd[ky]] = val
            line = fp.readline()
            linecount += 1

            if n is not None and linecount == n:
                break

        lengthd[split] = linecount


    X_train = X_train.tocsr()
    X_test  = X_test.tocsr()

    # Choose non-overlapping features
    if max_features is not None:
        counts  = array(X_train.astype(bool).astype(int).sum(axis=0)).ravel()
        order   = argsort(counts).ravel()[::-1]
        no_skipped = 0
        i    = 0
        inxs    = []
        hashset = set()
        while len(inxs) < max_features and i < len(counts):
            x     = X_train[:, order[i]].astype(float)
            hx    = md5hash(bytearray(x.todense()))
            if hx not in hashset:
                inxs.append(order[i])
                hashset.add(hx)
            else:
                no_skipped += 1
            i += 1

        X_train = X_train[:, inxs]
        X_test  = X_test[:, inxs]
        labelsd = dict(map(lambda i: (inxs.index(i), labelsd[i]), inxs))

    return  {"data": X_train, "target": y_train,
             "data_test": X_test, "target_test": y_test,
             "labels": labelsd}

# Wrappers for instances of the dataset
def load_books(n=None, max_features=None):
    return load_blitzer(dataset="books", n=n, max_features=max_features)

def load_dvd(n=None, max_features=None):
    return load_blitzer(dataset="dvd", n=n, max_features=max_features)

def load_kitchen(n=None, max_features=None):
    return load_blitzer(dataset="kitchen", n=n, max_features=max_features)

def load_electronics(n=None, max_features=None):
    return load_blitzer(dataset="electronics", n=n, max_features=max_features)










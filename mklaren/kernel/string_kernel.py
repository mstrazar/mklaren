import numpy as np
from itertools import product, combinations


# Kernel constanes
SPECTRUM          = "1spectrum"
SPECTRUM_MISMATCH = "2spectrum_mismatch"
WD                = "3weighted_degree_kernel"
WD_PI             = "4weighted_degree_kernel_pos_inv"
EXPONENTIAL_SPECTRUM = "5exponential_spectrum"


# Assume object are sequences
# or set of sequences
def spectrum_kernel(x1, x2, K=4, beacon=None, bin=None):
    """
    :param x1:
        Sequence of characters.
    :param x2:
        Sequence of characters.
    :param K:
        K-mers to be scanned.
    :param beacon:
        Beacon sequence (tuple of characters).
        If set, K is equal to beacon length and only beacons are counted.
    :param bin
        tuple (bin, number of all bins)
        Run kernel only in specified bin.
        Make sure sequences are of equal length!
    :return:
        Gram matrix.
    """
    if isinstance(beacon, str):
            beacon = tuple(beacon)
    K = len(beacon) if beacon else K
    kmers_i = zip(*[x1[k:] for k in range(K)])
    kmers_j = zip(*[x2[k:] for k in range(K)])
    if bin:
        assert len(x1) == len(x2)
        b, b_all = bin
        start = int(float(b)/b_all * len(kmers_i))
        end = int(float(b+1)/b_all * len(kmers_j))
        kmers_i = kmers_i[start:end]
        kmers_j = kmers_j[start:end]
    bin_norm = float(len(kmers_i)) if bin else 1
    if isinstance(beacon, type(None)):
        return np.sum([kmers_i.count(kmer)*kmers_j.count(kmer) for kmer in set(kmers_i) & set(kmers_j)]) / bin_norm
    else:
        return kmers_i.count(beacon) * kmers_j.count(beacon) / bin_norm


def spectrum_mismatch(x1, x2, K=4, m=1, bin=None):
    """
    :param x1:
        Sequence of characters.
    :param x2:
        Sequence of characters.
    :param K:
        K-mers to be scanned.
    :param bin
        tuple (bin, number of all bins)
        Run kernel only in specified bin.
        Make sure sequences are of equal length!
    :return:
        Gram matrix.
    """
    no_mismatches = lambda ki, kj: sum([not k1 == k2 for k1, k2 in zip(ki, kj)])

    # Return number of matches
    kmers_i = zip(*[x1[k:] for k in range(K)])
    kmers_j = zip(*[x2[k:] for k in range(K)])
    if bin:
        assert len(x1) == len(x2)
        b, b_all = bin
        start = int(float(b)/b_all * len(kmers_i))
        end = int(float(b+1)/b_all * len(kmers_j))
        kmers_i = kmers_i[start:end]
        kmers_j = kmers_j[start:end]
    bin_norm = float(len(kmers_i)) if bin else 1
    return np.sum([no_mismatches(ki, kj) < 2*m for ki, kj in product(kmers_i, kmers_j)]) / bin_norm


def weighted_degree_kernel(x1, x2, K=4, bin=None, beta=None, minK=2):
    """
    :param x1:
        Sequence of characters.
    :param x2:
        Sequence of characters.
    :param K:
        K-mers to be scanned.
    :param beta
        Weigth for different pairs of matches.
    :param bin
        tuple (bin, number of all bins)
        Run kernel only in specified bin.
        Make sure sequences are of equal length!
    :return:
        Gram matrix.
    """

    G = 0
    if bin:
        assert len(x1) == len(x2)
        b, b_all = bin

    for Kt in range(minK, K + 1):
        kmers_i = zip(*[x1[k:] for k in range(Kt)])
        kmers_j = zip(*[x2[k:] for k in range(Kt)])
        if bin:
            start = int(float(b)/b_all * len(kmers_i))
            end = int(float(b+1)/b_all * len(kmers_j))
            kmers_i = kmers_i[start:end]
            kmers_j = kmers_j[start:end]
        bin_norm = float(len(kmers_i)) if bin else 1
        g = np.sum([ki == kj for ki, kj in zip(kmers_i, kmers_j)]) / bin_norm
        if beta is None:
            beta = 2.0 * (K - Kt + 1) / (Kt * (Kt + 1))
        G += beta * g
    return G


def weighted_degree_kernel_pos_inv(x1, x2, K=4, var=8, beacon=None, bin=None):
    """
    Weighted degree kernel with positional invariance

    :param x1:
        Sequence of characters.
    :param x2:
        Sequence of characters.
    :param K:
        K-mers to be scanned.
    :param beacon:
        Beacon sequence (tuple of characters).
        If set, K is equal to beacon length and only beacons are counted.
    :param bin
        tuple (bin, number of all bins)
        Run kernel only in specified bin.
        Make sure sequences are of equal length!
    :return:
        Gram matrix.
    """
    G = 0
    if bin:
        assert len(x1) == len(x2)
        b, b_all = bin
    if not isinstance(beacon, type(None)):
        K = len(beacon)
        if isinstance(beacon, str):
                beacon = tuple(beacon)
    for Kt in range(2, K + 1):
        g = 0
        kmers_i = zip(*[x1[k:] for k in range(Kt)])
        kmers_j = zip(*[x2[k:] for k in range(Kt)])
        if bin:
            start = int(float(b)/b_all * len(kmers_i))
            end = int(float(b+1)/b_all * len(kmers_j))
            kmers_i = kmers_i[start:end]
            kmers_j = kmers_j[start:end]
        bin_norm = float(len(kmers_i)) if bin else 1

        for s in range(var):
            delta = 1.0 / (2*(s+1))
            if isinstance(beacon, type(None)):
                mu_i = np.sum([ki == kj for ki, kj in zip(kmers_i, kmers_j[s:])])
                mu_j = np.sum([ki == kj for ki, kj in zip(kmers_j, kmers_i[s:])])
                g += delta * (mu_i + mu_j)
            else:
                if Kt != len(beacon):
                    continue
                else:
                    mu_i = np.sum([beacon == ki == kj for ki, kj in zip(kmers_i, kmers_j[s:])])
                    mu_j = np.sum([beacon == ki == kj for ki, kj in zip(kmers_j, kmers_i[s:])])
                    g += delta * (mu_i + mu_j)
        beta = 2.0 * (K - Kt + 1) / (Kt * (Kt + 1.0)) / bin_norm
        G += beta * g
    return G



# Assume object are sequences
# or set of sequences
def exponential_spectrum(x1, x2, K=4, l=1):
    """
    Exponential string kernel. Applicable to strings of same length.

    :param x1:
        Sequence of characters.
    :param x2:
        Sequence of characters.
    :param K:
        K-mers to be scanned.
    :param l:
        Lengthscale parameter.
    :return:
        Kernel value.
    """
    # import matplotlib.pyplot as plt
    assert len(x1) == len(x2)

    # K-mer content
    krow1 = list(enumerate(zip(*[x1[i:] for i in range(K)])))
    krow2 = list(enumerate(zip(*[x2[i:] for i in range(K)])))

    # K-mer sets
    kset1 = set(map(lambda t: t[1], krow1))
    kset2 = set(map(lambda t: t[1], krow2))
    kint = kset1 & kset2

    kdata1 = dict()
    kdata2 = dict()

    # Compute kmer probability distributions
    N = len(krow1)
    for krow, kdata in (krow1, kdata1), (krow2, kdata2):
        for i, kmer in krow:
            if kmer not in kint: continue
    
            if kmer not in kdata:
                kdata[kmer] = np.zeros((N, ))
    
            t   = np.arange(0, N) - i
            vec = kdata[kmer]
            vec += np.exp(-t**2 / float(l))
            vec  = vec / vec.sum()
            kdata[kmer] = vec

    # Compute correlation between probability distributions
    k = 0
    for ky in kdata1.iterkeys():
        vec1 = kdata1[ky]
        vec2 = kdata2[ky]
        k += vec1.dot(vec2)

    return k



string_kernel_dict = {
    "1spectrum": spectrum_kernel,
    "2spectrum_mismatch": spectrum_mismatch,
    "3weighted_degree_kernel": weighted_degree_kernel,
    "4weighted_degree_kernel_pos_inv": weighted_degree_kernel_pos_inv,
    "5exponential_spectrum": exponential_spectrum,
}

# General wrapper
def string_kernel(X1, X2, mode="1spectrum", **kwargs):
    global string_kernel_dict
    if isinstance(X1, str): X1 = [X1]
    if isinstance(X2, str): X2 = [X2]
    f = string_kernel_dict[mode]
    G = np.zeros((len(X1), len(X2)))
    if id(X1) == id(X2):
        for (i, xi), (j, xj) in combinations(enumerate(X1), 2):
            G[j, i] = G[i, j] = f(xi, xj, **kwargs)
        for (i, xi) in enumerate(X1):
            G[i, i] = f(xi, xi, **kwargs)
    else:
        for (i, xi), (j, xj) in product(enumerate(X1), enumerate(X2)):
            G[i, j] = f(xi, xj, **kwargs)
    return G
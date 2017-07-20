import numpy as np

def generate_sequence(priors, L):
    """
    Generate a random sequence.

    :param priors:
        Prior probabilities of keys.
    :param L:
        Length of a sequence.
    :return:
    """
    keys = priors.keys()
    csum = np.cumsum([priors[ky] for ky in keys])
    seq = ""
    for l in range(L):
        seq += keys[np.argmax(np.random.rand() < csum)]
    return seq

def insert_motif(seq, motif, mean=0, var=3):
    """
     Insert a motif to sequences.

    :param seq:
        Initial sequence.
    :param motif:
        Motif to insert.
    :param mean:
        Mean position around center.
    :param var:
        Positional variance.
    :return:
        String with inserted motif.
    """
    j = int(len(seq)/2 + np.random.normal(mean, var))
    return (seq[:j] + motif + seq[j+len(motif):])[:len(seq)]

def generate_data(N, L, p, motif, mean, var,
                  priors={"A": 0.25, "C": 0.25, "G": 0.25, "T":0.25}, seed=None):
    """
    Generate random sequences with motif.

    :param N:
        Number of sequences.
    :param L:
        Length of sequences.
    :param p:
        Probability of motif insertion (positive class).
    :param motif:
        Motif to insert.
    :param mean:
        Mean potisioning.
    :param var:
        Positional variance
    :param priors:
        Prior probability of insertion.
    :param seed:
        Random seed.
    :return:
        data
            List of sequences.
        y
            Classes.
    """

    if seed is not None:
        np.random.seed(seed)
    data, y = list(), list()
    for n in range(N):
        seq = generate_sequence(priors, L)
        c = int(np.random.rand() < p)
        if c:
            seq = insert_motif(seq, motif, mean=mean, var=var)
        data.append(seq)
        y.append(c)
    y = np.array(y)
    return data, y

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from itertools import product
from collections import Counter


def coef_dict(p, d):
    """
    Construct a coefficient dictionary.
    :param p:
        Number of features.
    :param d:
        Degree of polynomial.
    :return:
    """

    # A sum is represented by integers
    t = tuple(range(p))
    coefs = dict()

    for dim in range(d+1):
        # Sort the bags to make them comparable
        terms = map(sorted, product(*dim*[t]))

        # Convert bags to indicator vectors
        bag2vec = lambda b: tuple([b.count(j) for j in range(p)])
        counts = dict(Counter(map(bag2vec, terms)))
        coefs.update(counts)

    return coefs


class Features(PolynomialFeatures):

    """
    Extend the ScikitLearn class of polynomial features
    to normalize the features by the correct exponents, such that the
    dot products in the new space will equal the values of correctly chose polynomial kernels.
    """

    def fit_transform(self, X, y=None, **fit_params):
        """
        Multiply by coefficients to obtain correct polynomial extension.

        :param X:
            Original features.
        :param y:
            Original signal.
        :param fit_params:
            Parameters to PolynomialFeatures.fit_transform
        :return:
        """
        Z = super(Features, self).fit_transform(X, y, **fit_params)
        pows = self.powers_

        self.coef_dict = coef_dict(p=X.shape[1], d=self.degree)
        self.coefs = np.array([np.sqrt(self.coef_dict[tuple(pw)]) for pw in pows])

        return Z * self.coefs
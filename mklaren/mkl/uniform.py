from numpy import ndarray, ones

class UniformAlignment:

    """
        Uniform kernel alignment.

    """
    def __init__(self):
        self.trained = False

    def fit(self, Ks):
        """
        :param Ks:
            List of kernel interfaces.
        """
        self.mu = ones((len(Ks), ))
        self.Ks = Ks
        self.trained = True

    def __call__(self, i, j):
        assert self.trained
        if isinstance(i, ndarray):
            i = i.astype(int).ravel()
        if isinstance(j, ndarray):
            j = j.astype(int).ravel()
        if isinstance(i, int) and isinstance(j, int):
            return sum([K[i, j] for K in self.Ks])
        else:
            return sum([K[i, :][:, j] for K in self.Ks])

    def __getitem__(self, item):
        assert self.trained
        return sum([K[item] for K in self.Ks])



class UniformAlignmentLowRank(UniformAlignment):
    """
        Uniform kernel alignment when kernels are represented by low-rank
        approximations.
    """

    def __call__(self, i, j):
        assert self.trained
        if isinstance(i, ndarray):
            i = i.astype(int).ravel()
        if isinstance(j, ndarray):
            j = j.astype(int).ravel()
        return sum([G[i, :].dot(G[j, :].T) for G in self.Ks])


    def __getitem__(self, item):
        assert self.trained
        return sum([G[item[0]].dot(G[item[1]].T) for G in self.Ks])

# from oct2py import octave
from os.path import join, dirname, realpath
import numpy as np



class CSI:

    """
    Cholesky with side information.
    Bach & Jordan, ICML 2005
    """

    def __init__(self, rank=40, kappa=0.99, centering=1, delta=40, eps=1e-4,):
        """
        :param rank:
            Maximal decomposition rank.
        :param kappa:
            Trade-off accuracy vs. predictive gain.
        :param centering:
            Centering of the kernel matrix.
        :param delta:
            No. of look-ahead columns.
        :param eps:
            Tolerance lower bound.
        """
        self.rank      = rank
        self.delta     = delta
        self.kappa     = kappa
        self.centering = centering
        self.tol       = eps
        self.I         = list()
        self.active_set_ = list()
        self.trained   = False
        # octave.addpath(join(dirname(realpath(__file__)), 'csi'))


    def fit(self, K, y):
        """
             INPUT
             K  : kernel matrix n x n
             y  : target vector n x d
             m  : maximal rank
             kappa : trade-off between approximation of K and prediction
                of Y (suggested: .99)
             centering : 1 if centering, 0 otherwise (suggested: 1)
             delta : number of columns of cholesky performed
                in advance (suggested: 40)
             tol : minimum gain at iteration (suggested: 1e-4)

             OUTPUT
             G : Cholesky decomposition -> K(P,P) is approximated by G*G'
             P : permutation matrix
             Q,R : QR decomposition of G (or center(G) if centering)
             error1 : tr(K-G*G')/tr(K) at each step of the decomposition
             error2 : ||Y-Q*Q'*Y||_F^2 / ||Y||_F^2 at each step
                of the decomposition
             predicted_gain : predicted gain before adding each column
             true_gain : actual gain after adding each column

             Copyright (c) Francis R. Bach, 2005.
        """

        # Convert to explicit form
        K = K[:, :]
        y = y.reshape((len(y), 1))

        # Call original implementation
        G, P, Q, R, error1, error2, error, predicted_gain, true_gain \
            = octave.csi(K, y, self.rank, self.centering, self.kappa,
                         self.delta, self.tol)

        # Octave indexes from 1
        P = P.ravel().astype(int) - 1

        # Resort rows to respect the order
        n, k = G.shape
        self.I = self.active_set_= list(P[:k])

        Go = np.zeros((n, k))
        Qo = np.zeros((n, k))
        Go[P, :] = G[:, :k]
        Qo[P, :] = Q[:, :k]
        Ro = R[:k, :k]
        self.G = Go[:, :self.rank]
        self.P = P[:self.rank]
        self.Q = Qo[:, :]
        self.R = Ro[:, :self.rank]
        self.error1 = error1
        self.error1 = error2
        self.error  = error
        self.predicted_gain = predicted_gain
        self.true_gain = true_gain
        self.trained = True
        self.active_set_ = self.I[:self.rank]


    def __call__(self, i, j):
        assert self.trained
        return self.G[i, :].dot(self.G[j, :].T)


    def __getitem__(self, item):
        assert self.trained
        return self.G[item[0], :].dot(self.G[item[1], :].T)


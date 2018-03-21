import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv
from scipy.stats import multivariate_normal as mvn
from examples.lars.lars_beta import plot_path, plot_residuals
from examples.lars.cholesky import cholesky_steps
from examples.lars.qr import qr_steps, qr_reorder, qr_orient, solve_R
from examples.lars.lars_group import p_ri, p_const, p_sc
from warnings import warn
from mklaren.util.la import safe_divide as div
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import exponential_kernel

# TODO: debugging: maximum delta (changes at each step) should yield exact estimates
# TODO: retain order of Ks in fit and Xs in predict

hlp = """
    LARS with multiple kernels.
    Use a penalty function to influence the selection of new kernels.
"""


class LarsMKL:

    def __init__(self, rank, delta):
        self.rank = rank
        self.delta = delta
        self.Q = None
        self.R = None
        self.Ri = None
        self.P = None
        self.Acts = None
        self.Ks = None
        self.path = None
        self.mu = None
        self.T = None
        self.trained = False

    @staticmethod
    def zy_app(G, Q, y, QTY):
        """ Approximate |<z, y>|^2 for current approximations G, Q.
            QTY is cached for all kernels. """
        n = G.shape[0]
        if G.shape[1] == 0:
            return -np.inf * np.ones((n,))
        GTY = G.T.dot(y)
        GTQ = G.T.dot(Q)
        A1 = GTY.dot(GTY.T) - GTQ.dot(QTY.dot(GTY.T))
        C = np.array([G[i, :].T.dot(A1).dot(G[i, :]) for i in range(n)])
        A2 = G.T.dot(G) - GTQ.dot(GTQ.T)
        B = np.array([G[i, :].T.dot(A2).dot(G[i, :]) for i in range(n)])
        B = np.round(B, 5)
        return div(C, B)

    def fit(self, Ks, y, f=p_ri):
        """ LARS-MKL sampling algorithm with a penalty function. Produces a feature space. """
        assert all([isinstance(K, Kinterface) for K in Ks])
        rank = self.rank
        delta = self.delta

        if delta == 1:
            raise ValueError("Unstable selection of delta. (delta = 1)")

        # Shared variables
        k = len(Ks)
        n = Ks[0].shape[0]
        Gs = []
        Acts = []
        Inas = []

        # Master QR decomposition and index of kernels and Permutation
        Q = np.zeros((n, rank + k * delta))
        R = np.zeros((rank + k * delta, rank + k * delta))
        P = []

        # Current status and costs
        corr = np.zeros((k,))  # Current correlation per kernel || X.T y || (required for calculating gain)
        cost = np.zeros((k,))  # Current derived cost per kernel || X.T y || * f(p)
        ncol = np.zeros((k,))  # Current number of columns per kernel (p)
        gain = np.zeros((k, n))  # Individual column gains

        # Look-ahead phase
        for j in range(len(Ks)):
            Gs.append(np.zeros((n, rank + delta)))
            Acts.append([])
            Inas.append(range(n))

            # Initial look-ahead setup ; one step per kernel
            if delta > 0:
                cholesky_steps(Ks[j], Gs[j], act=[], ina=range(n), max_steps=delta)

        # Iterations to fill the active set
        for step in range(rank):

            # Compute gains
            Qa = Q[:, :step]
            QTY = Qa.T.dot(y)
            for j in range(k):
                if delta > 0:
                    Ga = Gs[j][:, step:step+delta]
                    zy2 = self.zy_app(Ga, Qa, y, QTY)
                    gain[j, :] = (zy2 > 0) * (zy2 * f(ncol[j] + 1) + (f(ncol[j] + 1) - f(ncol[j])) * corr[j])
                    gain[j, Acts[j]] = -np.inf
                else:
                    gain[j, :] = Ks[j].diagonal() - (Gs[j] ** 2).sum(axis=1).ravel()
                    gain[j, Acts[j]] = -np.inf

            # Select optimal kernel and pivot
            kern, pivot = np.unravel_index(np.argmax(gain), gain.shape)
            if gain[kern, pivot] <= 0:
                msg = "Iterations ended prematurely at step = %d < %d" % (step, rank)
                warn(msg)
                rank = step
                break
            assert pivot not in Acts[kern]

            # Select pivot and update Cholesky factors; try simplyfing
            G = Gs[kern]
            K = Ks[kern]
            P.append(kern)
            k_inx = np.where(np.array(P) == kern)[0]
            k_num = len(k_inx)
            k_start = k_num - 1

            # Update Cholesky
            G[:, k_start:] = 0
            cholesky_steps(K, G, start=k_start, act=Acts[kern], ina=Inas[kern], order=[pivot])
            qr_steps(G, Q, R, max_steps=1, start=k_start, qstart=step)
            assert norm(G[:, :k_num] - (Q.dot(R))[:, k_inx]) < 1e-3

            # Clear invalid columns and update lookahead steps;
            # Copy act/ina
            max_steps = min(delta, G.shape[1] - k_num)
            cholesky_steps(K, G,
                           act=list(Acts[kern]),
                           ina=list(set(range(n)) - set(Acts[kern])),
                           start=k_num,
                           max_steps=max_steps)
            assert norm(G[:, :k_num] - (Q.dot(R))[:, k_inx]) < 1e-3

            # Update current correlation and total cost
            ncol[kern] += 1
            corr[kern] = norm(Q[:, k_inx].T.dot(y)) ** 2
            cost[kern] = corr[kern] * f(ncol[kern])

        # Correct lars order is based on including groups ; Gs are no longer valid
        del Gs
        korder = np.argsort(-cost).astype(int)
        porder = []
        for j in korder:
            porder.extend(list(np.where(np.array(P) == j)[0]))
        qr_reorder(Q, R, rank, porder)
        qr_orient(Q, R, y)
        assert rank == len(porder)

        # Use reduced approximation and store
        self.Ks = [Ks[j] for j in korder]
        self.Acts = [Acts[j] for j in korder]
        self.Q = Q[:, :rank]
        self.R = R[:rank, :rank]
        # self.Ri = solve_R(self.R)
        self.Ri = inv(self.R)
        self.P = np.array(P)[porder]

        # Compute transform for prediction with final matrices
        d2 = np.atleast_2d
        A = np.vstack([inv(d2(K[a, a])).dot(d2(K[a, :]))
                       for K, a in zip(self.Ks, self.Acts)
                       if len(a)])
        self.T = A.dot(self.Q).dot(self.Ri.T).dot(self.Ri)
        self.trained = True
        return

    def path(self, y, f=p_ri):
        """ Compute the group LARS path. """
        assert self.trained
        Q, R, P = self.Q, self.R, self.P
        korder = sorted(set(P), key=lambda p: np.argmax(P == p))
        pairs = zip(korder, korder[1:])
        path = np.zeros((len(korder), Q.shape[1]))
        t = np.sum(P == korder[0])
        r = y.ravel()
        mu = 0

        # Compute steps
        for i, (k1, k2) in enumerate(pairs):
            p1 = sum(P == k1)
            p2 = sum(P == k2)
            c1 = norm(Q[:, P == k1].T.dot(r))**2 * f(p1)
            c2 = norm(Q[:, P == k2].T.dot(r))**2 * f(p2)
            assert c2 <= c1
            alpha = 1 - np.sqrt(c2 / c1)
            path[i] = alpha * Q.T.dot(r).ravel()
            t += np.sum(P == k2)
            r = r - Q[:, :t].dot(path[i])
            mu = mu + Q[:, :t].dot(path[i])

        # Jump to least-squares solution
        path[-1] = Q.T.dot(r).ravel()
        r = r - Q.dot(path[-1])
        mu = mu + Q.dot(path[-1])
        assert norm(Q.T.dot(r)) < 1e-3
        self.path = path
        self.mu = mu
        return

    def transform(self, Xs):
        """ Map samples in Xs to the Q-space. """
        Ka = np.hstack([K(X, K.data[act, :])
                        for X, K, act in zip(Xs, self.Ks, self.Acts)
                        if len(act)])
        return Ka.dot(self.T)

    def predict(self, Xs):
        """ Predict values for Xs for the points in the active sets. """
        assert self.trained
        assert self.path is not None
        return self.transform(Xs).dot(self.path[-1]).ravel()


# Unit tests
def test_transform():
    """ Validity of predictive transform (idpt. of penalty). """
    noise = 0.04
    n = 100
    X = np.linspace(-10, 10, n).reshape((n, 1))
    Ks = [
        Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 1.0}),  # short
        Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.1}),  # long
    ]
    Kt = 0.00 + Ks[0][:, :] + 1.0 * Ks[1][:, :]
    for i in range(1000):
        f = mvn.rvs(mean=np.zeros(n, ), cov=Kt).reshape((n, 1))
        y = mvn.rvs(mean=f.ravel(), cov=noise * np.eye(n)).reshape((n, 1))
        model = LarsMKL(rank=20, delta=5)
        model.fit(Ks, y, p_const)
        Qt = np.round(model.transform([X, X]), 5)
        Q = np.round(model.Q, 5)
        assert norm(Q-Qt) < 1e-3



def compare_penalty():
    """ Simple test for LARS-kernel. """
    noise = 0.04
    n = 100
    X = np.linspace(-10, 10, n).reshape((n, 1))
    Ks = [
        Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 1.0})[:, :],    # short
        Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": 0.1})[:, :],    # long
        ]
    Kt = 0.00 + Ks[0] + 1.0 * Ks[1]
    f = mvn.rvs(mean=np.zeros(n,), cov=Kt).reshape((n, 1))
    y = mvn.rvs(mean=f.ravel(), cov=noise * np.eye(n)).reshape((n, 1))

    # Fit models
    # Q_ri, R_ri, P_ri = mkl_qr(Ks, y, rank, delta, f=p_ri)
    # path_ri, mu_ri = mkl_lars(Q_ri, P_ri, y, p_ri)
    #
    # Q_co, R_co, P_co = mkl_qr(Ks, y, rank, delta, f=p_const)
    # path_co, mu_co = mkl_lars(Q_co, P_co, y, p_const)
    #
    # Q_sc, R_sc, P_sc = mkl_qr(Ks, y, rank, delta, f=p_sc)
    # path_sc, mu_sc = mkl_lars(Q_sc, P_sc, y, p_sc)
    #
    # plt.figure()
    # plt.plot(y, ".")
    # plt.plot(f, "--")
    # plt.plot(mu_co, "-", label="unscaled", color="gray")
    # plt.plot(mu_sc, "-", label="scaled", color="pink")
    # plt.plot(mu_ri, "-", label="rich", color="green")
    # plt.legend()



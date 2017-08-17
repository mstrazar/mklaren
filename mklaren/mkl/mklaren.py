"""
The Mklaren algorithm (Multiple kernel learning with least-angle regression) perform simultaneous low-rank approximation of multiple kernel learning using least-angle regression.

    M. Strazar and T. Curk, "Learning the kernel matrix via predictive low-rank approximations", arXiv Prepr. arXiv1601.04366, 2016.


Given :math:`p` kernel matrices :math:`\mathbf{K}_1, \mathbf{K}_2, ..., \mathbf{K}_p`,  and a vector of targets :math:`\mathbf{y} \in \mathbb{R}^n`,
learn Cholesky factors :math:`\mathbf{G}_1, \mathbf{G}_2, ..., \mathbf{G}_p`, and a regression line :math:`\mathbf{\mu} \in \mathbb{R}^n`.

The regression line is learned in the union of normalized spans of Cholesky factors using least-angle regression.
The main advantage of the approach is the efficiency due to simultaneous MKL and low-rank approximations. The computational complexity is

.. math ::
    O(K^3 + pnK^2 \delta^2)
"""

from ..util.la import safe_divide as div, outer_product, safe_func, qr
from ..kernel.kinterface import Kinterface
from ..projection.nystrom import Nystrom
from numpy import zeros, diag, sqrt, mean, argmax, \
    array, log, eye, ones, absolute, ndarray, where, sign, min as npmin, argmin,\
    sum as npsum, isnan, isinf, vstack
from numpy.linalg import inv, norm
from numpy.random import choice

ssqrt = lambda x: safe_func(x, f=sqrt, val=0)

class Mklaren:
    """
    :ivar beta: (``numpy.ndarray``) Regression coefficients.

    :ivar G: (``numpy.ndarray``) Stacked Cholesky factors.

    :ivar G_mask: (``numpy.ndarray``) Indicator array of kernel order in ``G``.

    :ivar Ks: (``Kinterface``) Kernels in the model.

    :ivar bias: (``float``) Intercept term.

    :ivar regr: (``numpy.ndarray``) Regression line.
    """

    def __init__(self, rank, lbd=0, delta=10, debug=False):
        """
        :param rank: (``int``) Maximum allowed rank of the combined feature space.

        :param lbd: (``float``) L2 (ridge) regularization parameter.

        :param delta: (``int``)  Number of look-ahead steps.

        :param debug: (``bool``) Display debugging information.

        """
        self.rank      = rank
        self.lbd       = lbd
        self.delta     = delta
        self.trained   = False
        self.debug     = debug
        self.nystrom   = None
        self.sol_path  = []
        assert self.lbd >= 0


    def fit(self, Ks, y):
        """
        Learn low-rank approximations and regression line for kernel matrices or Kinterfaces.

        :param Ks: (``list``) of (``numpy.ndarray``) or of (``Kinterface``) to be aligned.

        :param y: (``numpy.ndarray``) Regression targets.
        """

        # Reshape y
        if len(y.shape) == 1:
            y = y.reshape((len(y), 1))

        # Decomposition parameters (including regulariation)
        n           = self.n = Ks[0].shape[0]
        n_reg       = n + self.rank
        no_kernels  = len(Ks)           # Number of kernels
        rank        = self.rank         # Total maximum rank
        delta       = self.delta        # Look-ahead steps for each kernel
        cols        = delta + rank      # Maximum number of Chol. columns per K
        lbd         = self.lbd          # Ridge regularization

        # Regression parameters
        bias        = mean(y)
        residual    = vstack([y - bias, zeros((rank, 1))])
        regr        = zeros((n_reg, 1))

        bisec       = None
        A           = None

        # Addressed by the kernel number
        data = dict()
        for pi in xrange(no_kernels):
            data[pi] = dict()
            data[pi]["K"]    = Ks[pi]               # Kernel matrices /functions
            data[pi]["G"]    = zeros((n, cols))     # Cholesky factors
            data[pi]["D"]    = Ks[pi].diag() if isinstance(Ks[pi], Kinterface) \
                else diag(Ks[pi]).copy()            # Diagonal not touched
                                                    # by lambda
            data[pi]["act"]  = list()               # active (pivot) set
            data[pi]["ina"]  = range(n)             # inactive set
            data[pi]["k"]    = 0                    # column count


        # Perform lookahead Cholesky steps for each kernel
        if self.debug:
            print "=== Start of the lookahead steps ==="

        for pi in xrange(no_kernels):
            # Pass by reference
            G = data[pi]["G"]
            K = data[pi]["K"]
            D = data[pi]["D"]
            act = data[pi]["act"]
            ina = data[pi]["ina"]
            G, D, act, ina = self.cholesky_steps(K=K, G=G, D=D, start=0, act=act,
                                                 ina=ina, no_steps=self.delta,)

        # Reset active and inactive sets
        for pi in xrange(no_kernels):
            K = data[pi]["K"]
            data[pi]["lookahead"] = data[pi]["act"][:]      # look-ahead set cp
            data[pi]["act"]       = []                      # active set
            data[pi]["ina"]       = range(n)                # inactive set
            data[pi]["k"]         = 0                       # column count
            data[pi]["D"]   = K.diag() if isinstance(K, Kinterface) \
                else diag(K).copy()
            assert len(data[pi]["lookahead"]) == self.delta

        # Start main loop
        if self.debug:
            print "=== Start of main loop ==="

        # Global input space and list of kernels corresponding to each column
        # Note the columns have been augmented and the corresponding transform
        # is stored
        Xa          = zeros((n_reg, rank))
        Xa_mask     = zeros((rank, ))
        xbar_outer  = zeros((rank, ))
        xnorm_outer = zeros((rank, ))
        gbar_inner  = zeros((rank, ))
        gnorm_inner = zeros((rank, ))

        # Iterate until reaching maximum rank if the regression input space
        for xk in xrange(rank):

            # Select best (kernel, pivot) based on correlation with the
            # residual
            # Selected using the LAR criterion for gradient, where gradient
            # information is discarded and is re-calculated when the actual
            # for of Cholesky column becomes known.
            best_pi, best_i = self.best_pivot_kernel(data=data,
                                                       Xa=Xa[:, :xk+1],
                                                       residual=residual,
                                                       bisec=bisec,
                                                       A=A,
                                                       lbd=lbd,
                                                       xk=xk)
            if self.debug:
                print "Best kernel, pivot: %d, %d" % (best_pi, best_i)

            # Load data for selected kernel
            # Make sure this is passed as a reference (updated in the process)
            pi        = best_pi
            i         = best_i
            G         = data[pi]["G"]
            D         = data[pi]["D"]
            K         = data[pi]["K"]
            k         = data[pi]["k"]
            act       = data[pi]["act"]
            ina       = data[pi]["ina"]
            lookahead = data[pi]["lookahead"]
            assert best_i in ina
            assert best_i not in act

            # If pivot column in lookahead, select another and add to lookahead
            # based on standard gain criterion. Else, add it to the k-th pos.
            reorder = lookahead[k:]
            assert len(reorder) == len(set(reorder))
            if i in reorder:
                inaj    = list(set(ina) - set(reorder) - set([i]))
                reorder.remove(i)
                j       = inaj[argmax(D[inaj])]
                reorder = [i] + reorder + [j]
                assert j in ina
                assert j != i
            else:
                reorder = [i] + reorder
            lookahead = lookahead[:k] + reorder
            data[pi]["lookahead"] = lookahead

            # 2. Naively repeat the Cholesky steps
            #    for the reordered columns
            D_reorder   = D.copy()
            G, _, _, _  = self.cholesky_steps(K=K, G=G, D=D_reorder,
                                             start=k,
                                             act=act[:],
                                             ina=ina[:],
                                             order=reorder,)

            # Active inactive modified
            act.append(i)
            ina.remove(i)

            # Modify original D at the currently selected pivot only
            D[ina] = D[ina] - (G[ina, k]**2)
            D[i]   = 0

            # Add newly created column to regression input space
            # Store transform for individual g
            g                   = G[:, k]
            gbar_inner[xk]      = g.mean()
            gnorm_inner[xk]     = norm(g - g.mean())

            # Store global input space and transform
            # Note the double standardization
            Xa[:n,   xk]    = self.standardize(g)
            Xa[n+xk, xk]    = sqrt(lbd)
            Xa[:, xk]       = Xa[:, xk] * 1.0/sqrt(lbd + 1.0)
            Xa_mask[xk]     = pi

            # Get gradient via LAR equation and ignore pivot
            # Update only at the second step!
            if xk > 0:
                # Calculate actual c and a of newly added column
                gn = Xa[:, xk]
                c_vec = array([gn.dot(residual)])
                a_vec = array([gn.dot(bisec)])

                grad, _ = self.gradient(X=Xa[:, :xk], bisec=bisec,
                                     residual=residual, A = A, ina=[0],
                                     a_vec=a_vec, c_vec=c_vec)

                # Update regression line and residual
                regr     = regr + grad * bisec
                residual = residual - grad * bisec
                self.sol_path.append(regr[:-self.rank])

                # Update maximum correlation
                C  = max(absolute(Xa.T.dot(residual)))

            # New bisector
            bisec, A = self.bisector(X=Xa[:, :xk+1], residual=residual)

            # Compute last step
            if xk == rank - 1:
                grad = C / A
                regr = regr + grad * bisec
                residual = residual - grad * bisec
                self.sol_path.append(regr[:-self.rank])

            # Increase column counter
            data[pi]["k"] = k + 1


        # Recover beta when both transforms are applied
        Q, R            = qr(Xa)
        self.beta       = inv(R).dot(Q.T.dot(regr))
        self.coef_      = self.beta

        # Recover data from inner and outer transform
        Xa = Xa * sqrt(1.0 + lbd)
        Xa = Xa * gnorm_inner
        Xa = Xa + gbar_inner

        # Store Cholesky factors and kernels (required for Tny
        # ction)
        # Address with true n now to forget about lambdas
        # Decode also the inner transform
        self.gbar       = gbar_inner
        self.gnorm      = gnorm_inner
        self.G          = Xa[:n, :]
        self.G_mask     = Xa_mask
        self.Ks         = Ks
        self.trained    = True
        self.regr       = regr[:n]
        self.bias       = bias
        self.y_pred     = bias + self.regr
        for pi in xrange(no_kernels):
            data[pi]["G"] = data[pi]["G"][:, :data[pi]["k"]]
            assert data[pi]["k"] == len(data[pi]["act"])

        for pi in data.keys():
            cols = where(Xa_mask == pi)[0]
            if len(cols):
                data[pi]["beta"]  = self.beta[cols]
                data[pi]["gbar"]  = self.gbar[cols]
                data[pi]["gnorm"] = self.gnorm[cols]
                data[pi]["xbar"]  = xbar_outer[cols]
                data[pi]["xnorm"] = xnorm_outer[cols]
                k = data[pi]["k"]

                if self.debug:
                    assert norm(data[pi]["G"][:, :k] - self.G[:, cols]) < 1e-5

        # Approximate kernel weights as the sum of coefficient absolute values
        self.mu = zeros((len(Ks),))
        for ki, K in enumerate(Ks):
            inxs = where(Xa_mask == ki)[0]
            if len(inxs):
                self.mu[ki] = norm(self.G[:,inxs].dot(self.beta[inxs]))

        # Fit Nystrom method to selected pivots to obtain kxk operators
        #   to Cholesky factor spaces
        self.data = data
        self.fit_nystrom()


    def cholesky_steps(self, K, G, D, start, act, ina, order=None,
                       no_steps=None,):
        """
        Perform Cholesky steps for kernel K, starting from the existing matrix
        G at index k. Order of newly added pivots may be specified.

        :param K:
            Kernel matrix / interface.
        :param G:
            Pre-existing Cholesky factors.
        :param D:
            Existing diagonal.
        :param k:
            Starting index.
        :param order:
            Possible to specify desired order.
            If not specified, standard gain criterion is taken.
        :param no_steps.
            Number of steps to take.
        :return:
            Updated Cholesky factors.
        """
        if order is None:
            no_steps   = K.shape[0] if no_steps is None else no_steps
            have_order = False
        else:
            no_steps    = len(order)
            have_order  = True

        for ki, k in enumerate(xrange(start, start + no_steps)):

            # Select best pivot
            i = order[ki] if have_order else ina[argmax(D[ina])]
            act.append(i)
            ina.remove(i)

            # Perform Cholesky step for the selected pivot
            j       = list(ina)
            G[:, k] = 0
            G[i, k] = ssqrt(D[i])
            G[j, k] = div(1.0, G[i, k]) * \
                      (K[j, i] - G[j, :k].dot(G[i, :k].T))

            # Store previous and update diagonal
            D[j] = D[j] - (G[j, k]**2)
            D[i] = 0

            # Debug step
            if self.debug:
                assert norm(K[:, :]-G[:, :k].dot(G[:, :k].T)) >= \
                       norm(K[:, :]-G[:, :k+1].dot(G[:, :k+1].T))

        return G, D, act, ina

    def standardize(self, X):
        """
        Standardize X.

        :param X:
            Original, non-standardized matrix.
        :return:
            A standardized copy of array X.
        """
        Y    = X.copy()
        ybar = Y.mean(axis=0)
        Y    = div(Y - ybar, norm(Y - ybar, axis=0))
        return Y


    def best_pivot_kernel(self, data, Xa, residual, bisec, A, lbd=0, xk=0,
                          tol=1e-12):
        """
        Select best kernel, pivot pair. Determined by
            a) pivot with *maximum* correlation with the residual (step 0)
            b) pivot with a *minimum* gradient such that a new variable will join
               the active set (step 1, ..., rank)
        :param step
            Step of the global decomposition to determine placement of reg.
            coefficients.
        :return:
            pi: kernel number
            i: pivot number
        """
        step         = 0 if bisec is None else True
        best_kernel  = 0
        best_val     = 0 if step == 0 else float("inf")
        best_pivot   = 0

        for pi in data.keys():
            ina     = data[pi]["ina"]
            act     = data[pi]["act"]
            G       = data[pi]["G"]
            k       = data[pi]["k"]       # k is the index for NEXT column
            kadv    = min(k+self.delta, G.shape[0])
            Ga      = G[:, k:kadv]  # Pass look ahead G to evaluate gains
            c_vec, a_vec = self.gain(G=Ga, residual=residual, bisec=bisec,
                                     active=act, lbd=lbd, step=xk)
            c_vec = c_vec[ina]
            a_vec = a_vec[ina] if bisec is not None else None

            # Exclude zeros correlations
            if self.debug:
                print "Kernel:%d, step:%d, max:%f" % (pi, k, max(c_vec))
            if max(absolute(c_vec)) > tol:
                gamma, pivot = self.gradient(X=Xa, bisec=bisec,
                                             A=A, residual=residual,
                                             c_vec=c_vec,
                                             a_vec=a_vec, ina=ina)

                # Apply different criterion for steps a/b
                if (step == 0 and gamma > best_val) or \
                        (step > 0 and gamma < best_val):
                    best_val     = gamma
                    best_kernel  = pi
                    best_pivot   = pivot
                    assert best_pivot in ina

        return best_kernel, best_pivot


    def bisector(self, X, residual):
        """
        Get current bisector on the vectors in X.
        Assume the input are already of correct dimensions.

        :param X:
            Matrix of covariates.
        :param xbar:
            Vector of covariate means.
        :param residual:
            Regression residual.
        :return:
            A: maximum correlation with the bisector.
            bisec: bisector.
        """

        # Standardize covariates
        n      = X.shape[0]

        # In the direction of the residual
        # Do so only for data rows
        xsig   = sign(X.T.dot(residual)).ravel()
        X      = X * xsig

        # Return normalized vector
        k = X.shape[1]
        if k == 1:
            X      = X[:, :1]
            CX     = X.T.dot(X)         # pair-wise projection
            CX_inv = inv(CX)            # solving linear system
            k_ones = ones((k, 1))       # ones vector
            A      = (k_ones.T.dot(CX_inv).dot(k_ones)) ** -0.5
            bisec  = X[:, 0:1]
        else:
            # Calculate new bisector
            CX     = X.T.dot(X)         # pair-wise projection
            CX_inv = inv(CX)            # solving linear system
            k_ones = ones((k, 1))       # ones vector
            A      = (k_ones.T.dot(CX_inv).dot(k_ones)) ** -0.5
                                                        # active set corr.
                                                        # with the bisector
                                                        # (normalizing const.)
            w      = A * CX_inv.dot(k_ones)             # auxilliary
            bisec  = X.dot(w)                           # equiangular vector

        if self.debug:
            assert absolute(norm(bisec) - 1.0) < 1e-5
            assert norm(X.T.dot(bisec) - A) < 1e-5
        return bisec, A


    def gradient(self, X, bisec, A, residual, c_vec, a_vec, ina):
        """
        The function is used in two contexts:
            a) Selection of the candidate pivot based in vector of correlations
                with residual (c_vec) and bisector (a_vec) from ina. This is
                happening within a G for one kernel.
            b) Determination of gradient, given that candidate pivot has already
                been selected (ina is set of length 1). This is happening within
                the global X for all kernels.

        Assume the input are already of correct dimensions.

        :param X:
            Matrix of covariates.
        :param bisec:
            Bisector vector.
        :param c_vec:
            Estimated correlations with the residual.
        :param a_vec:
            Estimated correlations with the bisector.
        :param ina
            Inactive set to index c_vec, a_vec.
        :param single
            Select minimal absolute value, do not discard negatives. Used when
            repairing the regression line when new ("random") column is added.
        :return:
            gamma: gradient
            pivot: pivot index
        """
        assert len(c_vec) == len(ina)
        assert (a_vec is None) or (len(a_vec) == len(c_vec))

        if bisec is None:
            return max(c_vec), ina[argmax(c_vec)]

        xsig    = sign(X.T.dot(residual)).ravel()
        X       = X * xsig
        C       = max(absolute(X.T.dot(residual)))

        # Get minimum over positive components
        # Since X is standardized, this always exists.
        scores = zeros((2, len(ina)))
        scores[0, :] = div(C - c_vec, A - a_vec).ravel()
        scores[1, :] = div(C + c_vec, A + a_vec).ravel()
        scores[where(scores == 0)] = float("inf")
        scores = npmin(scores, axis=0)
        gamma  = npmin(scores)
        pivot  = ina[argmin(scores)]
        assert not isinf(gamma)
        return gamma, pivot


    def gain(self, G, residual, bisec, active, lbd=0, step=0):
        """
        Calculate the LAR gain for the remaining columns given
        the current approximation and the residual. It is calculated for
        each kernel individually.

        Apply the projection matrix (I - 11^T/n) to translate columns
        and to ensure gradient exists at each step.

        Assume the input are already of correct dimensions.

        :param X:
            Cholesky factors including only look-ahead columns.
            Size is always n.
        :param residual:
            Regression residual.
            Size depends on whether regularization is included (n or n+k).
        :param bisec:
            Current bisector.
            Size depends on whether regularization is included (n or n+k).
        :param active:
            Active set for the current kernel.
        :return
            a_vec:
                Estimated correlations with the bisector or None.
            c_vec:
                Estimated correlations with the residual.
        """

        # Regularization constants
        # Note: lf cancels out in all fractions!
        # Note: lc is zero in dot products (for bisec and residual),
        # since the corresponding position
        # has not yet been updated by any previous bisector
        l = sqrt(lbd)
        lf = 1.0 # 1.0 / sqrt(1 + lbd)

        # Norm of lookeahead columns
        n    = G.shape[0]
        o    = ones((n, 1))
        GTG  = G.T.dot(G)
        G1   = G.T.dot(o)
        GM   = GTG - G1.dot(G1.T)/n
        lc   = lbd * (1.0 - 1.0 / (n + self.rank))
        A1   = ssqrt(lc + array([G[i, :].dot(GM).dot(G[i, :].T)
                            for i in xrange(n)])).ravel()

        # Correlation with the residual
        v       = residual
        v_std   = v - v.mean()
        vn_std  = v_std[:n] - v_std[:n].mean()
        A2      = G.dot(G.T.dot(vn_std)).ravel()
        A2      = A2 + sqrt(lbd) * v_std[n + step]
        c_vec   = div(A2, A1).ravel()
        if len(active):
            c_vec[active] = 0

        # Correlation with the bisector
        if bisec is not None:
            v       = bisec
            v_std   = v - v.mean()
            vn_std  = v_std[:n] - v_std[:n].mean()
            A3      = G.dot(G.T.dot(vn_std)).ravel()
            A3      = A3 + sqrt(lbd) * v_std[n + step]
            a_vec   = div(A3, A1)
            if len(active):
                a_vec[active] = 0
        else:
            return c_vec, None

        # Alignment cost debug
        if self.debug:
            Ladv = G.dot(G.T)
            k    = self.rank
            P    = eye(n, n)     - 1.0 * ones((n, n))/n
            Pk   = eye(n+k, n+k) - 1.0 * ones((n+k, n+k))/(n+k)
            for i in xrange(G.shape[0]):
                lv = zeros((k, 1))
                lv[step] = l
                gi = Pk.dot(lf * vstack([P.dot(Ladv[:, i:i+1]),
                                         lv]))
                assert norm(norm(gi)           - lf * A1[i]) < 1e-3
                assert norm(gi.T.dot(residual) - lf * A2[i]) < 1e-3
                assert norm(gi.T.dot(bisec)    - lf * A3[i]) < 1e-3

        return c_vec, a_vec


    def fit_nystrom(self):
        """
            Fit kernels using the Nystrom method given the predefined indices.

            Assume all kernel are given by list of Kinterfaces.

            Add mappings to self.data
                Mapping kernel_index -> (kernel, kernel (XS), kernel(SS)^1 )
                where S is th active set.
        """
        for pi in self.data.keys():
            K    = self.data[pi]["K"]
            G    = self.data[pi]["G"]
            inxs = self.data[pi]["act"]
            if len(inxs):
                nystrom = Nystrom()
                nystrom.fit(K, inxs=inxs)
                Kssi    = nystrom.K_SS_i
                Kxs     = nystrom.K_XS
                self.data[pi]["Kxs"] = Kxs
                self.data[pi]["Kss"] = Kssi
                self.data[pi]["Tny"] = inv(G.T.dot(G)).dot(G.T).\
                    dot(Kxs).dot(Kssi)


    def predict(self, Xs, Ks=None):
        """Predict responses for test samples.

        Each of the kernel low rank approximation has got its corresponding
        primal regression coefficients stored.

        :param Xs: (``list``) of (``numpy.ndarray``) Input space representation for each kernel in ``self.Ks``.

        :param Ks: (``list``) of (``numpy.ndarray``) Values of the kernel against K[test set, training set]. Optional.

        :return: (``numpy.ndarray``) Vector of prediction of regression targets.
        """
        assert self.trained
        assert (Xs is not None and len(Xs) == len(self.data)) or \
               (Ks is not None and len(Ks) == len(self.data))

        nt = Ks[0].shape[0] if Ks is not None else Xs[0].shape[0]
        regr = zeros((nt, 1)) + self.bias

        for pi in sorted(self.data.keys()):
            if "Tny" in self.data[pi]:
                K    = self.data[pi]["K"]
                Tny  = self.data[pi]["Tny"]
                inxs = self.data[pi]["act"]
                beta = self.data[pi]["beta"]
                gbar = self.data[pi]["gbar"]
                gnorm = self.data[pi]["gnorm"]
                if Ks is None:
                    Kts = K(Xs[pi], K.data[inxs])
                else:
                    Kts = Ks[pi][:, inxs]
                if len(Kts.shape) == 1:
                    Kts = Kts.reshape((Kts.shape[0], 1))
                if len(Tny.shape) == 1:
                    Tny = Tny.reshape((Tny.shape[0], 1))
                Gt   = Tny.dot(Kts.T).T
                Gt   = (Gt - gbar) / gnorm
                Gt   = Gt * 1.0/sqrt(self.lbd + 1.0)
                regr = regr + Gt.dot(beta)
        return regr.ravel()


    def __call__(self, i, j):
        """
        Access portions of the combined kernel matrix at indices i, j.

        :param i: (``int``) or (``numpy.ndarray``) Index/indices of data points(s).

        :param j: (``int``) or (``numpy.ndarray``) Index/indices of data points(s).

        :return:  (``numpy.ndarray``) Value of the kernel matrix for i, j.
        """
        assert self.trained
        if isinstance(i, ndarray):
            i = i.astype(int).ravel()
        if isinstance(j, ndarray):
            j = j.astype(int).ravel()
        return self.G[i, 1:].dot(self.G[j, 1:].T)


    def __getitem__(self, item):
        """
        Access portions of the kernel matrix generated by ``kernel``.

        :param item: (``tuple``) pair of: indices or list of indices or (``numpy.ndarray``) or (``slice``) to address portions of the kernel matrix.

        :return:  (``numpy.ndarray``) Value of the kernel matrix for item.
        """
        assert self.trained
        return self.G[item[0], 1:].dot(self.G[item[1], 1:].T)

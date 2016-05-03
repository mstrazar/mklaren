from ..kernel.kinterface import Kinterface
from sklearn.kernel_approximation import Nystroem
from numpy import array
from numpy.linalg import inv
from numpy.random import choice
from scipy.linalg import sqrtm

class Nystrom:

    """
        Standard Nystrom method implementation.
        Inducing inputs can be given or selected randomly.
    """


    def __init__(self, rank=10):
        """
        :return:
        """
        self.trained = False
        self.rank = rank


    def fit(self, K, inxs=None):
        """
        Fit approximation to the kernel function / matrix.

        :param K:
            Kernel matrix / kernel interface.
        :param inxs:
            Inducing inputs. If None, select randomly, else use specified inputs
            and override ranks.
        """
        self.n       = K.shape[0]
        if inxs is None:
            inxs = choice(xrange(self.n),
                          size=self.rank, replace=False)
        self.rank    = len(inxs)
        self.K       = K
        self.active_set_  = inxs
        self.K_SS    = K[inxs, inxs]
        self.K_XS    = K[:, inxs]
        if len(inxs) > 1:
            self.K_SS_i  = inv(K[inxs, :][:, inxs])
        else:
            self.K_SS_i  = array([[1.0 / K[inxs[0], inxs[0]]],])
        R = sqrtm(self.K_SS_i)
        self.G = self.K_XS.dot(R)
        self.trained = True


    def predict(self, X=None, inxs=None):
        """
        :param X:
            Test samples in the input space.
        :param inxs
            Indices in the training data.
        :return:
            Predicted values of the kernel function against all training
            inputs.
        """
        if X is None and inxs is None:
            # Full approximation
            return self.K_XS.dot(self.K_SS_i).dot(self.K_XS.T)
        elif X is not None:
            # Invoke kernel directly
            K_YS = self.K.kernel(X, self.K.data[self.active_set_, :])
        elif inxs is not None:
            # Read existing values
            K_YS = self.K[inxs, self.active_set_]
        K_YS = K_YS.dot(self.K_SS_i).dot(self.K_XS.T)
        return K_YS


class NystromScikit:

    """
        Nystrom implementation form Scikit Learn wrapper.
        The main difference is in selection of inducing inputs.
    """

    def __init__(self, rank=10, random_state=42):
        """
        :param rank:
        :return:
        """
        self.trained = False
        self.rank = rank
        self.random_state = random_state


    def fit(self, K, y):
        """
        Fit approximation to the kernel function / matrix.

        :param K:
            Kernel matrix / kernel interface.
        :param y:
            Target matrix.
        """
        assert isinstance(K, Kinterface)

        self.n           = K.shape[0]
        kernel           = lambda x, y: K.kernel(x, y, **K.kernel_args)
        self.model       = Nystroem(kernel=kernel,
                                    n_components=self.rank,
                                    random_state=self.random_state)

        self.model.fit(K.data, y)
        self.active_set_ = list(self.model.component_indices_[:self.rank])
        assert len(set(self.active_set_)) == len(self.active_set_) == self.rank
        R = self.model.normalization_
        self.G = K[:, self.active_set_].dot(R)
        self.trained = True








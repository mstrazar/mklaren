from mklaren.kernel.kernel import poly_kernel
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank
from sklearn.metrics import mean_squared_error
import numpy as np

# TODO: enable prediction with sums of kernels in CSI for comparison

# Fixed hyper parameters
delta = 10
repeats = 10
range_n = [30, 100, 300]
range_degree = range(2, 6)
methods = ["Mklaren", "CSI"]
rank = 5   # Rank is known because the data is simulated
p_tr = 0.75
p_te = 1.0 - p_tr
P = 1   # Number of true kernels to be taken in the sum
lbd = 0

n = 30
maxd = 2

# Training / test split
tr = range(int(n * p_tr))
te = range(tr[-1]+1, n)

X = np.random.rand(n, rank)
X = (X - X.mean(axis=0)) / np.std(X, axis=0)

X_tr = X[tr]
X_te = X[te]

Ks_tr = []
Ks_all = []
for d in range(1, maxd + 1):
    K_tr = Kinterface(kernel=poly_kernel, kernel_args={"degree": d}, data=X_tr)
    K_a = Kinterface(kernel=poly_kernel, kernel_args={"degree": d}, data=X)
    Ks_tr.append(K_tr)
    Ks_all.append(K_a)

mu_true = np.zeros((len(Ks_tr),))
mu_true[-P] = 1

# True kernel matrix to generate the signal ;
# weights are defined only by the training set
alpha = np.random.rand(n, 1)
alpha[te] = 0
K_true = sum([mu_true[i] * Ks_all[i][:, :] for i in range(len(Ks_all))])
y_true = K_true.dot(alpha)

for method in methods:

    # Fit the mklaren method and predict
    if method == "Mklaren":
        model_mklaren = Mklaren(rank=len(Ks_tr) * rank, delta=delta, lbd=lbd)
        model_mklaren.fit(Ks_tr, y_true[tr])
        y_pred = model_mklaren.predict([X_te] * len(Ks_tr))
        y_fit = model_mklaren.predict([X_tr] * len(Ks_tr))
        w_fit = model_mklaren.mu / model_mklaren.mu.sum()
    elif method == "CSI":
        model_csi = RidgeLowRank(rank=rank, method="csi", lbd=lbd,
                                 method_init_args={"delta": delta})
        model_csi.fit(Ks_tr, y_true[tr])
        y_pred = model_csi.predict([X_te] * len(Ks_tr))
        y_fit = model_csi.predict([X_tr] * len(Ks_tr))

    # Score the predictions
    mse_pred = mean_squared_error(y_true[te], y_pred)
    mse_fit = mean_squared_error(y_true[tr], y_fit)
    print(method, mse_pred, mse_fit)
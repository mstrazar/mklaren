from mklaren.mkl.mklaren import Mklaren
from mklaren.projection.csi import CSI
from mklaren.projection.icd import ICD
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import numpy as np


# Do pivots tend to represent some portions of space better than others, or is the
# relation purely random?

# Does this transfer to the PSD cone of kernels?
n = 1000
rank = 2
P = 2

alpha = np.random.rand(n, 1).ravel()

# Basically sum of rank one kernels, tu can be arbitrarily big.
Ks = []
for p in range(P * 5):
    X = np.random.rand(n, rank)
    if p < P:
        # Features are very small
        # X = X / 1000
        X = np.power(X, 4)
    K = X.dot(X.T)
    Ks.append(K)

K_true = sum(Ks[:P])
K_all = sum(Ks)
y = K_true.dot(alpha)
y = y - y.mean()
mu_true = np.zeros((len(Ks), ))
mu_true[:P] = 1

# Mklaren
model = Mklaren(rank=P * rank)
model.fit(Ks, y)
y_pred = model.regr

rho, pv = spearmanr(mu_true, model.mu)
rho_fit, pv_fit = spearmanr(y_pred, y)
p_rho_fit, p_pv_fit = pearsonr(y_pred.ravel(), y.ravel())
mse = np.linalg.norm(y_pred - y)
print("Mklaren weights: %f, %f" % (rho, pv))
print("Mklaren fit (s): %f, %f" % (rho_fit, pv_fit))
print("Mklaren fit (p): %f, %f" % (p_rho_fit, p_pv_fit))
print("Mklaren mse: %f" % mse)


# CSI
csi = CSI(rank=P * rank)
csi.fit(K_all, y)
lin_model = LinearRegression()
lin_model.fit(csi.G, y)
y_pred = lin_model.predict(csi.G)
rho_fit, pv_fit = spearmanr(y_pred, y)
p_rho_fit, p_pv_fit = pearsonr(y_pred.ravel(), y.ravel())
mse = np.linalg.norm(y_pred - y)
print("CSI fit (s): %f, %f" % (rho_fit, pv_fit))
print("CSI fit (p): %f, %f" % (p_rho_fit, p_pv_fit))
print("CSI mse: %f" % mse)
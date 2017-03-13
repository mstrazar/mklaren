import numpy as np
from numpy.linalg import inv, norm, solve
from mklaren.projection.icd import ICD

n = 20 # full rank
p = 3
P = 3   # Number of kernels

mu = np.random.rand(P, 1).ravel()
alpha = np.random.rand(n, 1)

# Generate kernels
Ks = []
Gs = []
for i in range(P):
    Gs.append(np.random.rand(n, p))
    Ks.append(Gs[i].dot(Gs[i].T))

# Combined kernel matrix and feature space
L = sum([mu[i] * Ks[i] for i in range(P)])
H = np.hstack([np.sqrt(mu[i]) * Gs[i] for i in range(P)])
assert np.round(norm(L-H.dot(H.T)), 3) == 0

# Stacked Gs (without the weights)
# This is actually visible by the algorithm and the weights get encoded into beta
S = np.hstack([Gs[i] for i in range(P)])
print("Stacked feature space size: %s" % str(S.shape))

# Generate visible data
y = L.dot(alpha)

# Min norm solution to beta (linear regression)
SiS = inv(S.T.dot(S))
print(SiS)
beta = SiS.dot(S.T.dot(y))
print "Beta app error:", norm(y - S.dot(beta))
assert np.round(norm(y -  S.dot(beta)), 3) == 0


# Compute weights by approximating alpha
alpha_app = S.dot(SiS.dot(beta))
print "Alpha app error:", norm(alpha-alpha_app)

Z = np.zeros((n, P))
for i in range(P):
    # Z[:, i] = Gs[i].dot(beta[i*p:(i+1)*p]).ravel()
    # Z[:, i] = Ks[i].dot(alpha).ravel()  # True solution
    Z[:, i] = Ks[i].dot(alpha_app).ravel()

mu_app = inv(Z.T.dot(Z)).dot(Z.T.dot(y)).ravel()


# Approximation: norm of contribution to the weight vector
mu_app = np.zeros((P, ))
for i in range(P):
    mu_app[i] = np.linalg.norm(Gs[i].dot(beta[i*p:(i+1)*p]).ravel())
mu_app /= norm(mu_app)

print("Mu (app): ", mu_app)
print("Mu", mu)
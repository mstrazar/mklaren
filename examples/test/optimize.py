hlp = """ Interactive;
          Maximizing the squared correlation with respect to the lengthscale
          for the exponentiated quadratic kernel. This way, the value of
          the function is independent of the potential coefficient to
          be given to the current inducing point.
      """

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn, pearsonr
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface


true_gamma = 1.5
n = 100
noise = 0.01
X = np.linspace(-10, 10, n).reshape((n, 1))
K = Kinterface(data=X, kernel=exponential_kernel, kernel_args={"gamma": true_gamma})
f = mvn.rvs(mean=np.zeros((n,)), cov=K[:, :] + noise * np.eye(n, n)).reshape((n, 1))

# Inducing point
xi = X[n/2, :].reshape((1, 1))
gamma_range = np.linspace(0.001, 5, 100)
errors = []
cors = []
sols = []
for gam in gamma_range:
    gi = exponential_kernel(X, xi, gamma=gam)
    error = np.linalg.norm(gi - f)
    cors.append(pearsonr(gi, f)[0]**2)
    sols.append(gi)
    errors.append(error)

plt.figure()
plt.plot(gamma_range, errors, "k-")
plt.plot(gamma_range, cors, "r-")

plt.figure()
gi = sols[np.argmax(cors)]
plt.plot(X.ravel(), f.ravel())
plt.plot(X.ravel(), gi.ravel())
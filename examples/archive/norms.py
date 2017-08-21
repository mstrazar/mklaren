# The norm of columns inevitably changes as the polynomial features
# are added. It is probably not possible to normalize the features via
# accessing the kernel only, but its possible to divide by the norm
# in the feature space. In any case, the contribution of and individual column
# diminishes as the interaction exponent grows.

from examples.features import Features
import numpy as np
import matplotlib.pyplot as plt

degree_range = range(1, 7)
n_range = [30, 100, 300]
p = 10
repeats = 10

data_norms = np.zeros((repeats, len(n_range), len(degree_range)))
data_means = np.zeros((repeats, len(n_range), len(degree_range)))


for repl in range(repeats):

    for ni, n in enumerate(n_range):

        # Generate a random, centered & normalized dataset
        X = np.random.rand(n, p)
        X = (X - X.mean(axis=0)) / np.std(X, axis=0)

        for di, d in enumerate(degree_range):
            feats = Features(degree=d)
            Phi = feats.fit_transform(X)

            # Normalize by the norm of feature space representation
            row_norms = np.linalg.norm(Phi, axis=1)
            row_norms = row_norms.reshape((n, 1))
            Phi = Phi / row_norms

            data_norms[repl, ni, di] = np.mean(np.linalg.norm(Phi, axis=0) / n)
            data_means[repl, ni, di] = np.mean(Phi)


# Plot results
norm_means = np.mean(data_norms, axis=0)
norm_std = np.std(data_norms, axis=0)
mean_means = np.mean(data_means, axis=0)
mean_std = np.std(data_means, axis=0)

# Norms
plt.figure()
for ni, n in enumerate(n_range):
    plt.errorbar(degree_range, norm_means[ni, :], yerr=norm_std[ni, :],
                 label="n=%d" % n, linewidth=1+ni)
plt.legend()
plt.xlabel("Polynomial degree")
plt.ylabel("Average column norm")
plt.xlim(0, degree_range[-1]+0.5)
plt.savefig("examples/output/norms/comparison_norms.pdf",
            bbox_inches="tight")

# Means
plt.figure()
for ni, n in enumerate(n_range):
    plt.errorbar(degree_range, mean_means[ni, :], yerr=mean_std[ni, :],
                 label="n=%d" % n, linewidth=1+ni)
plt.legend()
plt.xlabel("Polynomial degree")
plt.ylabel("Average cell value")
plt.xlim(0, degree_range[-1]+0.5)
plt.savefig("examples/output/norms/comparison_means.pdf",
            bbox_inches="tight")
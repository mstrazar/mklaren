hlp = """
    What is the magnitude of the weights learned by the different methods?
"""


from examples.snr.snr import *
from scipy.linalg import inv

n = 100
gamma_range = np.logspace(-3, 3, 7)

count = 0
for seed in range(30):
    Ksum, Klist, inxs, X, Xp, y, f = generate_data(n=n, rank=5, inducing_mode="uniform",
                                                   noise=1, gamma_range=gamma_range, seed=seed, input_dim=1)
    r = test(Ksum, Klist, inxs, X, Xp, y, f, lbd=0, methods=["CSI", "Mklaren"])


    # Matrix norms
    K_mkl_norm = np.linalg.norm(r["Mklaren"]["model"][:, :])
    G = r["CSI"]["model"].Gs[0]
    K_csi_norm = np.linalg.norm(G.dot(G.T))

    beta_mkl  = r["Mklaren"]["model"].beta
    beta_csi = r["CSI"]["model"].beta

    # plot_signal(X, Xp, y, f, models=r)
    mkl_norm = np.linalg.norm(beta_mkl)
    csi_norm = np.linalg.norm(beta_csi)
    count += int(mkl_norm < csi_norm)

    print("MKL beta: %.3f, K: %.3f" % (mkl_norm, K_mkl_norm))
    print("CSI beta: %.3f, K: %.3f" % (csi_norm, K_csi_norm))
    print
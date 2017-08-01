import os
import csv
import scipy.stats as st
import datetime
import itertools as it
from mklaren.kernel.string_kernel import *
from mklaren.kernel.string_util import *
from mklaren.kernel.kernel import kernel_sum
from mklaren.kernel.kinterface import Kinterface
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeLowRank

# List available kernels
args = [
    {"mode": SPECTRUM, "K": 2},
    {"mode": SPECTRUM, "K": 3},
    {"mode": SPECTRUM, "K": 4},
    {"mode": SPECTRUM, "K": 5},
    {"mode": WD, "K": 2},
    {"mode": WD, "K": 4},
    {"mode": WD, "K": 5},
    {"mode": WD_PI, "K": 2},
    {"mode": WD_PI, "K": 3},
    {"mode": WD_PI, "K": 4},
]

# Hyperparameters
methods = ["Mklaren", "CSI", "Nystrom", "ICD"]
lbd_range  = [0] + list(np.logspace(-5, 1, 7))  # Regularization parameter
p_range = [1, 3, 10]
iterations = range(30)
rank = 10
delta = 10
var = 10

# Fixed output
# Create output directory
d = datetime.datetime.now()
dname = os.path.join("..", "output", "string",
                     "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname): os.makedirs(dname)
rcnt = len(os.listdir(dname))
fname = os.path.join(dname, "results_%d.csv" % rcnt)
print("Writing to %s ..." % fname)

# Output
header = ["n", "method", "rank", "iteration", "lambda",
          "p", "evar_tr", "evar_va", "evar"]
fp = open(fname, "w", buffering=0)
writer = csv.DictWriter(fp, fieldnames=header)
writer.writeheader()

# Training test split
tr = range(0, 50)
va = range(50, 75)
te = range(75, 100)

# Generate random datasets and perform prediction
count = 0
seed = 0
for cv, num_k in it.product(iterations, p_range):
    print("Seed: %d, num. kernels: %d" % (cv, num_k))
    X, _ = generate_data(N=100, L=100, p=0.5, motif="TGTG", mean=0, var=var, seed=seed)
    Xa = np.array(X)
    X_tr = Xa[tr]
    X_va = Xa[va]
    X_te = Xa[te]

    # Individual kernels
    Ks_full = [Kinterface(kernel=string_kernel, data=X, kernel_args=arg) for arg in args]

    # Individual and sum of kernels
    Ks = [Kinterface(kernel=string_kernel, data=X_tr, kernel_args=arg) for arg in args]
    Ksum = Kinterface(data=X_tr, kernel=kernel_sum,
                      kernel_args={"kernels": [string_kernel] * len(args),
                                   "kernels_args": args})

    # Random num_k kernels are relevant and form the true covariance matrix
    kinxs = np.random.choice(range(len(Ks)), size=num_k, replace=False)
    inxs = np.array([np.random.choice(tr, size=rank, replace=False).ravel()
                    for ki in kinxs])

    # Different index set at each kernel
    Ca = 0
    for ki, inx in enumerate(kinxs):
        Ki =  Ks_full[inx][:, inxs[ki]]
        Kii = Ki[inxs[ki], :]
        Ca += Ki.dot(np.linalg.inv(Kii)).dot(Ki.T)

    # Target signal is a sample from a GP
    y = st.multivariate_normal.rvs(mean=np.zeros((len(X),)), cov=Ca)
    y_tr = y[tr]
    y_va = y[va]
    y_te = y[te]

    # Modeling
    for method in methods:
        for lbd in lbd_range:
            yt, yv, yp = None, None, None

            if method == "Mklaren":
                mkl = Mklaren(rank=rank, lbd=lbd, delta=delta)
                try:
                    mkl.fit(Ks, y_tr)
                    yt = mkl.predict([X_tr] * len(Ks))
                    yv = mkl.predict([X_va] * len(Ks))
                    yp = mkl.predict([X_te] * len(Ks))
                except Exception as e:
                    print(e)
                    continue
            else:
                if method == "CSI":
                    model = RidgeLowRank(rank=rank, method="csi",
                                         method_init_args={"delta": delta}, lbd=lbd)
                else:
                    model = RidgeLowRank(rank=rank, method=method.lower(), lbd=lbd)
                try:
                    model.fit([Ksum], y_tr)
                    yt = model.predict([X_tr])
                    yv = model.predict([X_va])
                    yp = model.predict([X_te])
                except Exception as e:
                    print(e)
                    continue

            # Evaluate explained varaince on the three sets
            evar_tr = (np.var(y_tr) - np.var(yt - y_tr)) / np.var(y_tr)
            evar_va = (np.var(y_va) - np.var(yv - y_va)) / np.var(y_va)
            evar    = (np.var(y_te) - np.var(yp - y_te)) / np.var(y_te)

            row = {"n": len(X), "method": method,
                   "rank": rank, "iteration": cv, "lambda": lbd,
                    "p": num_k, "evar_tr": evar_tr, "evar_va": evar_va, "evar": evar}

            writer.writerow(row)
            seed += 1
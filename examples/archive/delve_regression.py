"""
    Evaluation of low-rank kernel approximation methods
    with ridge regression on standard datasets.
"""
import examples.__config__

# Low-rank approximation methods
from mklaren.regression.ridge import RidgeLowRank
from mklaren.regression.ridge import RidgeMKL
from mklaren.mkl.mklaren import Mklaren

# Kernels
from mklaren.kernel.kernel import exponential_kernel
from mklaren.kernel.kinterface import Kinterface

# Datasets
from datasets.delve import load_abalone
from datasets.delve import load_boston
from datasets.delve import load_comp_activ
from datasets.delve import load_bank
from datasets.delve import load_pumadyn
from datasets.delve import load_kin

# Utils
from numpy import var, mean, logspace, where, log10
from numpy.linalg import norm
from random import shuffle, seed
from os.path import join
from sys import argv


# Datasets and options
# Load max. 1000 examples
n    = 1000
dset = "boston" if len(argv) == 1 else argv[1]

# Load datasets with at most n examples
datasets = {
    "boston":        (load_boston,     {"n": n,}),
    "abalone":       (load_abalone,    {"n": n,}),
    "comp":   (load_comp_activ, {"n": n,}),
    "bank":      (load_bank, {"typ": "8fm", "n": n,}),
    "pumadyn":   (load_pumadyn, {"typ": "8fm", "n": n,}),
    "kin":       (load_kin, {"typ": "8fm", "n": n,}),
}



# Hyperparameters
rank_range = range(2, 7, 2)
                                           # Rank range
lbd_range  = logspace(-3, 3, 7)            # Regularization parameter
gam_range  = logspace(-3, 3, 7, base=2)    # RBF kernel parameter
delta      = 10                            # Number of look-ahead columns
                                           # (CSI and mklaren)

# Method classes and fixed hyperparameters
methods = {
    # "CSI":         (RidgeLowRank, {}),
    # "CSI" :        (RidgeLowRank, {"method": "csi", "delta": delta}),
    "ICD" :        (RidgeLowRank, {"method": "icd"}),
    "Nystrom":     (RidgeLowRank, {"method": "nystrom"}),
    "Mklaren":     (Mklaren,      {"delta": delta}),
}


# CV hyper parameters
cv_iter         = 5     # Cross-validation iterations
training_size   = 0.6   # Training set
validation_size = 0.2   # Validation set (for fitting hyperparameters)
test_size       = 0.2   # Test set (for reporting scores)


load, load_args = datasets[dset]
data = load(**load_args)


# Load data and normalize columns
X = data["data"]
X = X - X.mean(axis=0)
nrm = norm(X, axis=0)
nrm[where(nrm==0)] = 1
X = X / nrm
y = data["target"]
y = y - mean(y)
n = len(X)


# Perform model cross-validation with internal parameter selection
for cv in range(cv_iter):

        # Split into training, validation, test sets.
        seed(cv)
        inxs = range(len(X))
        shuffle(inxs)
        n1 = int(training_size * n)
        n2 = int(validation_size * n)
        tr, tval, te = inxs[:n1], inxs[n1:n1+n2], inxs[n1+n2:]
        X_tr, X_val, X_te = X[tr], X[tval], X[te]
        y_tr, y_val, y_te = y[tr], y[tval], y[te]

        # Store kernel interfaces for training data
        Ks = [Kinterface(data=X_tr,
                         kernel=exponential_kernel,
                         kernel_args={"gamma": gam}) for gam in gam_range]


        output = "\t".join(["Method", "effective rank", "iteration", "lambda", "RMSE"])
        print(output)

        for mname, (Mclass, kwargs) in methods.items():

            # Fit hyperparameters on the validation set using the current rank
            for rank in rank_range:
                best_lambda = lbd_range[0]
                best_score  = float("inf")
                if mname == "Mklaren":
                        effective_rank = rank * len(Ks)
                else:
                    effective_rank = rank

                for lbd in lbd_range:
                    model = Mclass(lbd=lbd, rank=effective_rank, **kwargs)
                    model.fit(Ks, y_tr)
                    yp    = model.predict([X_val for k in Ks])


                    score = var(y_val - yp)**0.5
                    if score < best_score:
                        best_score = score
                        best_lambda = lbd

                # Fit model with best hyperparameters
                model = Mclass(lbd=best_lambda, rank=effective_rank, **kwargs)
                model.fit(Ks, y_tr)
                yp    = model.predict([X_te for k in Ks])
                score = var(y_te - yp)**0.5
                output = "\t".join(map(str, [mname, rank, cv, \
                                             best_lambda, score]))
                print(output)









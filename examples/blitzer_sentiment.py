import __config__

# Methods
from mklaren.mkl.mklaren import Mklaren
from mklaren.mkl.align import AlignLowRank, Align
from mklaren.regression.ridge import RidgeMKL

# Kernels
from mklaren.kernel.kernel import linear_kernel, center_kernel_low_rank
from mklaren.kernel.kinterface import Kinterface

# Datasets
from datasets.blitzer import load_books, load_dvd,\
    load_electronics, load_kitchen

# Utils
from numpy import logspace, array, vstack, log10, argsort, var
from random import shuffle, seed

# Dataset and number of samples
dset    = "books" if len(argv) == 1 else argv[1]
n       = None
max_features = 500
delta   = 1     # Number of look-ahead columns


# Hyperparameters
lbd_range    = list(logspace(-3, 3, 7))
center       = True
cv_iter      = 5      # Cross-validation iterations

# Methods and order
methods = {
    "Mklaren": (Mklaren,  {"delta": delta, "rank": 1}),
    "uniform": (RidgeMKL, {"method": "uniform", "low_rank": True}),
    "align":   (RidgeMKL, {"method": "align", "low_rank": True}),
    "alignf":  (RidgeMKL, {"method": "alignf", "low_rank": True}),
    "alignfc": (RidgeMKL, {"method": "alignfc", "low_rank": True})
}


# Datasets and options
datasets = {
    "books":        (load_books, {"n": n,       "max_features": max_features}),
    "kitchen":      (load_kitchen, {"n": n,     "max_features": max_features}),
    "dvd":          (load_dvd, {"n": n,         "max_features": max_features}),
    "electronics":  (load_electronics, {"n": n, "max_features": max_features}),
}


training_size   = 0.8   # Training set
validation_size = 0.2   # Validation set (for selecting hyperparameters, etc.)


load, load_args = datasets[dset]
data = load(**load_args)

# Data for storage
results_test     = dict()
results_training = dict()
modelsd          = dict()

# Training/test pools
ndata      = data["data"].shape[0]
ndata_test = data["data_test"].shape[0]

# Load data
X = vstack([array(data["data"].todense()),
            array(data["data_test"].todense())])
y1 = data["target"]
y2 = data["target_test"]
y1 = y1.reshape((len(y1), 1))
y2 = y2.reshape((len(y2), 1))
y  = vstack([y1 ,y2])
inxs_tr = range(0, ndata)
inxs_te = range(ndata, ndata + ndata_test)

print X.shape
# Center targets with *training* mean
y = y - y[inxs_tr].mean()
n = X.shape[0]

# Rank-1 kernels for Cortes methods
Vs = [X[:, i].reshape((n, 1)) for i in range(X.shape[1])]
if center:
    Vs = map(center_kernel_low_rank, Vs)

# Set rank range wrt. training size
rank_range = [10, 20, 40, 80, 160, ]

for cv in range(cv_iter):
        seed(cv)

        # Split into training, validation, test sets.
        shuffle(inxs_tr)
        n1 = int(training_size   * len(inxs_tr))

        tr,   tval,    te = inxs_tr[:n1], inxs_tr[n1:], inxs_te
        y_tr, y_val, y_te = y[tr], y[tval], y[te]

        V_tr  = [Kinterface(data=v[tr], kernel=linear_kernel) for v in Vs]
        V_val = [v[tval] for v in Vs]
        V_te  = [v[te] for v in Vs]


        output = "\t".join(["Method", "effective rank", "iteration", "lambda", "RMSE"])
        print(output)

        for rank in rank_range:

            # Select a subset of kernels for full-rank methods ; use align
            premodel = AlignLowRank()
            premodel.fit(Vs, y, holdout = tval + te)
            kernel_order = argsort(premodel.mu.ravel())[::-1]
            Vs_subset = [Vs[kernel_order[i]] for i in range(rank)]

            # Fit hyperparameters on the validation set using the current rank
            for mname, (Mclass, kwargs) in methods.items():

                best_lambda = lbd_range[0]
                best_score  = float("inf")
                if mname == "Mklaren":
                    kwargs["rank"] = rank

                for lbd in lbd_range:
                    model = Mclass(lbd=lbd, **kwargs)
                    if mname == "Mklaren":
                        model.fit(V_tr, y_tr)
                        yp    = model.predict(V_val).ravel()
                    else:
                        holdout = tval + te
                        model.fit(Vs_subset, y, holdout=holdout)
                        yp    = model.predict(tval).ravel()
                    score = var(y_val.ravel() - yp)**0.5
                    if score < best_score:
                        best_score = score
                        best_lambda = lbd

                # Fit model with best hyperparameters
                model = Mclass(lbd=best_lambda, **kwargs)
                if mname == "Mklaren":
                    model.fit(V_tr, y_tr)
                    yp    = model.predict(V_te).ravel()
                else:
                    holdout = tval + te
                    model.fit(Vs_subset, y, holdout=holdout)
                    yp    = model.predict(te).ravel()

                score = var(y_te.ravel() - yp)**0.5
                output = "\t".join(map(str, [mname, rank, cv, \
                                             best_lambda, score]))
                print(output)

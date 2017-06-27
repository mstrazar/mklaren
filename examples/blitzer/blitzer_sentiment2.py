import sys
import os
import csv
import datetime

# Methods
from mklaren.mkl.mklaren import Mklaren
from mklaren.mkl.align import AlignLowRank
from mklaren.regression.ridge import RidgeMKL

# Kernels
from mklaren.kernel.kernel import linear_kernel, center_kernel_low_rank
from mklaren.kernel.kinterface import Kinterface

# Datasets
from datasets.blitzer import load_books, load_dvd,\
    load_electronics, load_kitchen

# Utils
from numpy import logspace, array, vstack, argsort, var, absolute
from random import shuffle, seed

# Dataset and number of samples
argv = dict(enumerate(sys.argv))
dset    = argv.get(1, "books")
n       = None
max_features = 500
delta   = 1     # Number of look-ahead columns
lbd_range    = list(logspace(-3, 3, 7))
center       = True
cv_iter      = 5        # Cross-validation iterations
training_size   = 0.8   # Training set
validation_size = 0.2   # Validation set (for selecting hyperparameters, etc.)

# Set rank range wrt. training size
rank_range = [10, 20, 40, 80, 160, 320]

# Fixed output
# Create output directory
d = datetime.datetime.now()
dname = os.path.join("..", "output", "blitzer_sentiment",
                     "%d-%d-%d" % (d.year, d.month, d.day))
if not os.path.exists(dname): os.makedirs(dname)
rcnt = len(os.listdir(dname))
fname = os.path.join(dname, "results_%d.csv" % rcnt)

print("Running on dataset % s" % dset)
print("Writing output to %s" % fname)


# Output
header = ["dataset", "method", "rank", "iteration", "lambda", "RMSE"]
fp = open(fname, "w", buffering=0)
writer = csv.DictWriter(fp, fieldnames=header)
writer.writeheader()


# Methods and order
methods = {
    "Mklaren": (Mklaren,  {"delta": delta, "rank": 1}),
    "uniform": (RidgeMKL, {"method": "uniform", "low_rank": True}),
    "align":   (RidgeMKL, {"method": "align", "low_rank": True}),
    "alignf":  (RidgeMKL, {"method": "alignf", "low_rank": True}),
    "alignfc": (RidgeMKL, {"method": "alignfc", "low_rank": True}),
    "l2krr":   (RidgeMKL, {"method": "alignfc", "low_rank": True})

}


# Datasets and options
datasets = {
    "books":        (load_books, {"n": n,       "max_features": max_features}),
    "kitchen":      (load_kitchen, {"n": n,     "max_features": max_features}),
    "dvd":          (load_dvd, {"n": n,         "max_features": max_features}),
    "electronics":  (load_electronics, {"n": n, "max_features": max_features}),
}



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

# Center targets with *training* mean
y = y - y[inxs_tr].mean()
n = X.shape[0]

# Rank-1 kernels for Cortes methods
Vs = [X[:, i].reshape((n, 1)) for i in range(X.shape[1])]
if center:
    Vs = map(center_kernel_low_rank, Vs)

for cv in range(cv_iter):
    seed(cv)

    # Split into training, validation, test sets.
    shuffle(inxs_tr)
    n1 = int(training_size   * len(inxs_tr))

    tr,   tval,    te = inxs_tr[:n1], inxs_tr[n1:], inxs_te
    y_tr, y_val, y_te = y[tr], y[tval], y[te]

    Ks_tr  = [Kinterface(data=v[tr], kernel=linear_kernel) for v in Vs]
    V_val = [v[tval] for v in Vs]
    V_te  = [v[te] for v in Vs]

    for rank in rank_range:

        # Fit hyperparameters on the validation set using the current rank
        for mname, (Mclass, kwargs) in methods.items():

            best_lambda = lbd_range[0]
            best_score  = float("inf")
            if mname == "Mklaren":
                kwargs["rank"] = rank

            for lbd in lbd_range:
                model = Mclass(lbd=lbd, **kwargs)
                if mname == "Mklaren":
                    model.fit(Ks_tr, y_tr)
                    yp    = model.predict(V_val).ravel()
                else:
                    holdout = tval + te
                    model.fit(Vs, y, holdout=holdout)
                    kernel_order = argsort(absolute(model.mu))[::-1]
                    Vs_subset = [Vs[kernel_order[i]] for i in range(rank)]
                    model.fit(Vs_subset, y, holdout=holdout)
                    yp    = model.predict(tval).ravel()
                score = var(y_val.ravel() - yp)**0.5
                if score < best_score:
                    best_score = score
                    best_lambda = lbd

            # Fit model with best hyperparameters
            model = Mclass(lbd=best_lambda, **kwargs)
            if mname == "Mklaren":
                model.fit(Ks_tr, y_tr)
                yp    = model.predict(V_te).ravel()
            else:
                holdout = tval + te
                model.fit(Vs_subset, y, holdout=holdout)
                yp    = model.predict(te).ravel()

            score = var(y_te.ravel() - yp) ** 0.5
            row = {"dataset": dset,
                   "method": mname,
                   "rank": rank,
                   "iteration": cv,
                   "lambda": best_lambda,
                   "RMSE": score}
            writer.writerow(row)
hlp = """
    Evaluation of the multiple kernel learning methods on the Blitzer sentiment analysis (text minig regression)
    dataset.
"""

import os
import csv
import argparse
from mklaren.mkl.mklaren import Mklaren
from mklaren.regression.ridge import RidgeMKL
from mklaren.kernel.kernel import linear_kernel, center_kernel_low_rank
from mklaren.kernel.kinterface import Kinterface
from datasets.blitzer import load_books, load_dvd,\
    load_electronics, load_kitchen
from numpy import logspace, array, vstack, argsort, var, absolute, ravel
from random import shuffle, seed

# Experiment parameters
n               = None                          # Limit number of data points (n=None to use all available)
max_features    = 500                           # Maximum number of features
delta           = 1                             # Number of look-ahead columns
lbd_range       = list(logspace(-3, 3, 7))      # Regularization parameters
center          = True                          # Center kernels
cv_iter         = 5                             # Cross-validation iterations
training_size   = 0.8                           # Training set proportion
validation_size = 0.2                           # Validation set (selecting hyperparameters)

# Set rank range
rank_range = [10, 20, 40, 80, 160, 320]


# Methods and order
methods = {
    "Mklaren": (Mklaren,  {"delta": delta, "rank": 1}),
    "uniform": (RidgeMKL, {"method": "uniform", "low_rank": True}),
    "align":   (RidgeMKL, {"method": "align", "low_rank": True}),
    "alignf":  (RidgeMKL, {"method": "alignf", "low_rank": True}),
    "alignfc": (RidgeMKL, {"method": "alignfc", "low_rank": True}),
    "l2krr":   (RidgeMKL, {"method": "l2krr", "low_rank": True})
}

# Datasets and options
datasets = {
    "books":        (load_books, {"n": n,       "max_features": max_features}),
    "kitchen":      (load_kitchen, {"n": n,     "max_features": max_features}),
    "dvd":          (load_dvd, {"n": n,         "max_features": max_features}),
    "electronics":  (load_electronics, {"n": n, "max_features": max_features}),
}



def process(dataset, outdir):
    """
    Run experiments with epcified parameters.
    :param dataset: Dataset key.
    :param outdir: Output directory.
    :return:
    """

    # Create output directory
    if not os.path.exists(outdir): os.makedirs(outdir)
    fname = os.path.join(outdir, "%s.csv" % dataset)
    print("Running on dataset %s" % dataset)
    print("Writing output to %s" % fname)

    # Output
    header = ["dataset", "method", "rank", "iteration", "lambda", "RMSE"]
    fp = open(fname, "w", buffering=0)
    writer = csv.DictWriter(fp, fieldnames=header)
    writer.writeheader()

    load, load_args = datasets[dataset]
    data = load(**load_args)

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
                best_subset = None
                if mname == "Mklaren":
                    kwargs["rank"] = rank

                for lbd in lbd_range:
                    model = Mclass(lbd=lbd, **kwargs)
                    if mname == "Mklaren":
                        model.fit(Ks_tr, y_tr)
                        yp    = model.predict(V_val).ravel()
                        subset = None
                    else:
                        if mname == "l2krr":
                            kwargs["method_init_args"] = {"lbd": lbd, "lbd2": lbd}
                        holdout = tval + te
                        model.fit(Vs, y, holdout=holdout)
                        kernel_order = argsort(absolute(model.mu.ravel())[::-1])
                        Vs_subset = [Vs[kernel_order[i]] for i in range(rank)]
                        model.fit(Vs_subset, y, holdout=holdout)
                        yp    = model.predict(tval).ravel()
                        subset = Vs_subset
                    score = var(y_val.ravel() - yp)**0.5
                    if score < best_score:
                        best_score = score
                        best_lambda = lbd
                        best_subset = subset

                # Fit model with best hyperparameters
                if mname == "l2krr":
                    kwargs["method_init_args"] = {"lbd": best_lambda, "lbd2": best_lambda}
                model = Mclass(lbd=best_lambda, **kwargs)
                if mname == "Mklaren":
                    model.fit(Ks_tr, y_tr)
                    yp    = model.predict(V_te).ravel()
                else:
                    holdout = tval + te
                    model.fit(best_subset, y, holdout=holdout)
                    yp    = model.predict(te).ravel()

                score = var(y_te.ravel() - yp) ** 0.5
                row = {"dataset": dataset,
                       "method": mname,
                       "rank": rank,
                       "iteration": cv,
                       "lambda": best_lambda,
                       "RMSE": score}
                writer.writerow(row)


if __name__ == "__main__":
    # Input arguments
    parser = argparse.ArgumentParser(description=hlp)
    parser.add_argument("dataset", help="Dataset. One of {books, dvd, electronics, kitchen}.")
    parser.add_argument("output",  help="Output directory.")
    args = parser.parse_args()

    # Output directory
    data_set = args.dataset
    out_dir = args.output
    assert data_set in datasets.keys()
    process(data_set, out_dir)
import numpy as np
import csv
from os import path
from os.path import join, dirname, realpath

ENERGY_PATH = join(dirname(realpath(__file__)), "energy")

DAY_ORDER = {"mon": 0, "tue": 1, "wed": 2, "thu": 3,
             "fri": 4, "sat": 5, "sun": 6}


def load_energy(signal="T1", aggregate_by="week"):
    """
    Load energy consuption signals, organized as time series for weeks.
    :param aggregate_by:
        Variable over which to aggregate.
    :param signal:
        Which signal to load.
    :return:
    """

    fname = path.join(ENERGY_PATH, "data.csv")
    reader = csv.DictReader(open(fname))

    if aggregate_by == "week":
        col_ky = ("dow", "hour")
        f_conv = int
    else:
        raise ValueError("Aggregation by %s not implemented." % aggregate_by)

    labels = set()
    data = dict()
    for row in reader:
        ky = f_conv(row[aggregate_by])
        if ky not in data: data[ky] = dict()
        cky = tuple((row[k] for k in col_ky))
        labels.add(cky)
        assert cky not in data[ky]
        data[ky][cky] = row[signal]

    # Fill in data table
    labels = sorted(labels, key=lambda t: (DAY_ORDER[t[0]], t[1]))
    X = np.zeros((len(data), len(labels)))
    for i, (ky, df) in enumerate(sorted(data.items())):
        for li, lab in enumerate(labels):
            X[i, li] = df.get(lab, np.nan)

    return {"data": X,
            "labels": labels}
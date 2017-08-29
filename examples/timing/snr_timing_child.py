hlp = """
    Child process worakround to test the running time of CSI, as the Oct2Py presents problems for
    the multithreading module. Pickle module is used to transfer data in a very ad-hoc manner.
    Not to be called independently.
"""
import sys
sys.path.append("../..")
import os
print(os.getcwd())
import pickle
from examples.inducing_points.inducing_points import test
f_in = sys.argv[1]
f_out = sys.argv[2]
obj = pickle.load(open(f_in))
Ksum, Klist, inxs, X, Xp, y, f = obj
r = test(Ksum, Klist, inxs, X, Xp, y, f, methods=("CSI",), lbd=0.1)
pickle.dump(r, open(f_out, "w"), protocol=pickle.HIGHEST_PROTOCOL)
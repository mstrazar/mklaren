hlp = """
    Child process worakround to test the running time of CSI.
"""
import sys
sys.path.append("../..")
import os
print(os.getcwd())
import pickle
from examples.snr.snr import test
f_in = sys.argv[1]
f_out = sys.argv[2]
obj = pickle.load(open(f_in))
Ksum, Klist, inxs, X, Xp, y, f = obj
r = test(Ksum, Klist, inxs, X, Xp, y, f, methods=("CSI",), lbd=0.1)
pickle.dump(r, open(f_out, "w"))
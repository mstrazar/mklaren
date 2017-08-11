import numpy as np
import time
from examples.snr.snr import generate_data, test
from multiprocessing import Manager, Process





def wrapsf(Ksum, Klist, inxs, X, Xp, y, f, method, return_dict):
    """ Worker thread """
    r = test(Ksum, Klist, inxs, X, Xp, y, f, methods=(method,), lbd=0.01)
    return_dict[method] = r[method]["time"]
    # return_dict[method] = np.linalg.inv(Klist[0][:, :]).shape
    return


def process():
    manager = Manager()
    return_dict = manager.dict()

    n = 100
    rank = 5
    gamma_range = [0.1]
    limit = 10

    Ksum, Klist, inxs, X, Xp, y, f = generate_data(n=n,
                                                   rank=rank,
                                                   inducing_mode="uniform",
                                                   noise=1.0,
                                                   gamma_range=gamma_range,
                                                   input_dim=1)

    # Evaluate methods
    method = "l2krr"
    t1 = time.time()
    p = Process(target=wrapsf, name="test",
                args=(Ksum, Klist, inxs, X, Xp,
                      y, f, method, return_dict))
    p.start()
    p.join(timeout=limit)
    t1 = time.time() - t1
    print("Total time: %f" % t1)
    print(return_dict.items())



if __name__ == "__main__":
    process()
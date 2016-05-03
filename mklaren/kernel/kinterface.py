from numpy import array, atleast_2d, ndarray
from scipy.sparse import isspmatrix, csr_matrix

class Kinterface:
        """
        Interface to invoke kernels. Acts as a wrapper for a unified use of
        kernel functions/matrices by implementing __getitem__ and __call__().

        This is necessary for methods that only require access to parts
        of the kernel matrix.
        """

        def __init__(self, data, kernel, kernel_args={}, data_labels=None):
            """
            :param data:
                Pointer to the data in the original input space.
            :param kernel:
                Kernel function (callable).
            :param data_labels
                Data labels for kernel if available.
            :return:
            """
            self.data   = data
            self.kernel = kernel
            self.kernel_args = kernel_args
            try:
                self.shape  = (len(data), len(data))
            except TypeError:
                self.shape  = (data.shape[0], data.shape[0])
            self.data_labels = data_labels



        def __getitem__(self, item):
            """
                Generalize all items to iterables.
                Item must be a tuple of two items, which are transformed either
                to list of iterable.
            """
            assert isinstance(item, tuple)
            assert len(item) == 2
            args = [None, None]
            for oi, obj in enumerate(item):
                if isinstance(obj, int):
                    assert obj >= 0
                    args[oi] = self.data[obj]
                    if isinstance(args[oi], ndarray):
                        args[oi] = atleast_2d(args[oi])
                elif isinstance(obj, slice):
                    start = obj.start if obj.start is not None else 0
                    stop  = obj.stop if obj.stop is not None else self.shape[0]
                    assert start >= 0
                    assert stop >= 0
                    args[oi] = self.data[start:stop]
                elif isinstance(obj, xrange) or isinstance(obj, list) or \
                    isinstance(obj, ndarray):
                    if isinstance(obj, ndarray) and len(obj.shape) > 1:
                        # Assume items represent data not indices
                        args[oi] = obj
                    elif isinstance(self.data, list):
                        args[oi] = [self.data[int(o)] for o in obj]
                    else:
                        args[oi] = self.data[map(int, obj)]

                elif isspmatrix(obj):
                    args[oi] = obj

                else:
                    raise NotImplementedError("Unknown addressing type.")

            # Ravel if vector
            r = self.kernel(args[0], args[1], **self.kernel_args)
            if hasattr(r, "shape") and 1 in r.shape:
                r = r.ravel()
            return r

        def __call__(self, i, j):
            """
            Mimic a callable kernel function.
            :param i:
                Index.
            :param j:
                Index.
            :return:
                Value of the kernel.
            """
            return self[i, j]


        def diag(self):
            """
            :return
                Diagonal of the kernel.
            """
            return array([self[i, i] for i in xrange(self.shape[0])]).ravel()




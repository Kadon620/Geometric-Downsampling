import os
import sys
import cffi
import time
import threading
import numpy as np


class FuncThread(threading.Thread):
    """函数线程类"""
    
    def __init__(self, target, *args):
        threading.Thread.__init__(self)
        self._target = target
        self._args = args

    def run(self):
        return self._target(*self._args)


def Knn(X, N, D, n_neighbors, forest_size, subdivide_variance_size, leaf_number):
    """
    使用C++ DLL进行K近邻搜索
    """
    ffi = cffi.FFI()
    ffi.cdef()

    try:
        t1 = time.time()
        dllPath = os.path.join('F:/knnDll.dll')
        C = ffi.dlopen(dllPath)

        cffi_X1 = ffi.cast('double*', X.ctypes.data)

        neighbors_nn = np.zeros((N, n_neighbors), dtype=np.int32)
        distances_nn = np.zeros((N, n_neighbors), dtype=np.float64)

        cffi_neighbors_nn = ffi.cast('int*', neighbors_nn.ctypes.data)
        cffi_distances_nn = ffi.cast('double*', distances_nn.ctypes.data)

        t = FuncThread(C.knn, cffi_X1, N, D, n_neighbors, cffi_neighbors_nn, cffi_distances_nn, 
                       forest_size, subdivide_variance_size, leaf_number)

        t.daemon = True
        t.start()

        while t.is_alive():
            t.join(timeout=1.0)
            sys.stdout.flush()

        print("knn runtime = %f" % (time.time() - t1))
        return neighbors_nn, distances_nn

    except Exception as ex:
        print(ex)

    return [[], []]
import os

import numpy as np

#################################################################################################################################
# Code for utilities related Dask distributed arrays and np.memmap based array access


def np_mmap(path, dtype, shape, orig_path=None, fs=None):
    if os.path.exists(path):
        return np.memmap(
            filename=path,
            mode="r+",
            dtype=np.dtype(dtype),
            shape=tuple(shape),
            offset=offset,
        )
    else:
        return np.memmap(
            filename=path,
            mode="w+",
            dtype=np.dtype(dtype),
            shape=tuple(shape),
            offset=offset,
        )

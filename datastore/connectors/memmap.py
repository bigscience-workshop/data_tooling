import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir, os.path.pardir)))

######################################################################################
# MemmapSharded
class MemmapSharded(np.memmap):
    """
    Extension of numpy's memmap class which provides a view into
    multiple shards of memmap files that can be lazy loaded, from
    disk, over a network, etc.
    """
    def __new__(subtype, shards, dtype=np.float32,
                shape=None, fs=None):
        mode='r+' 
        offset=0
        order='C'



#################################################################################################################################
# Code for utilities related Dask distributed arrays and np.memmap based array access

def np_mmap(path, dtype, shape, orig_path=None, fs=None):
    if os.path.exists(path):
        return np.memmap(filename=path, mode="r+", dtype=np.dtype(dtype), shape=tuple(shape), offset=offset)
    else:
        return np.memmap(filename=path, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape), offset=offset)


def mmap_dask_array(path, shape, dtype, blocksize=1000000, shards=None, fs=None, shared_dir=None, cached_dir="/tmp/"):
    '''
    Create a Dask array from raw binary data in :code:`filename`
    by memory mapping.

    Returns
    -------

    dask.array.Array
        Dask array matching :code:`shape` and :code:`dtype`, backed by
        memory-mapped chunks.
    '''
    load = dask.delayed(np_mmap)
    chunks = []
    if shared_dir and path.startswith(shared_dir):
      orig_path = path
      path = path.replace(shared_dir, cached_dir)
    for index in range(0, shape[0], blocksize):
        # Truncate the last chunk if necessary
        chunk_size = min(blocksize, shape[0] - index)
        chunk = dask.array.from_delayed(
            load(
                path.replace(".mmap", "")+f"_{index}_{index + chunk_size}.mmap",
                shape=(chunk_size, ) + shape[1:],
                dtype=dtype, 
                orig_path=orig_path,
                fs=fs,
            ),
            shape=(chunk_size, ) + shape[1:],
            dtype=dtype
        )
        chunks.append(chunk)
    return da.concatenate(chunks, axis=0)
"""
x = mmap_dask_array(
    filename='testfile-50-50-100-100-float32.raw',
    shape=(50, 50, 100, 100),
    dtype=np.float32
)
"""

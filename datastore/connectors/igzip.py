import tempfile
from threading import Timer, Thread
from multiprocessing import Process
import subprocess
import requests
import multiprocessing
from filelock import UnixFileLock, FileLock
try:
  from megatron.data.indexed_dataset import MMapIndexedDataset
except:
  MMapIndexedDataset = None

import snorkel
from functools import partial
from snorkel.labeling.apply.core import BaseLFApplier, _FunctionCaller
from snorkel.labeling.apply.pandas import apply_lfs_to_data_point, rows_to_triplets

from datastore.utils.persisted_row_shards import *

######################################################################################
# Indexed Gzip files broken up into shards. Optimized for accessing
# efficiently various gzip files.

class IndexGzipFileSharded(PersistedRowShards):
    """An indexable structure made of multiple gzip files that are
    viewed as a single line indexed file. Each igzip is lazy loaded
    when needed.
    :arg shard_defs: A list of dict, with fields 'file', 'start', 'end' and optionaly 'fobj'.
    :arg mode: either 'r' or 'rb
    :arg fs: the fsspec file system or None for local os file system.
    :arg chunksize: the chunk size to use when iterating.
    """
    def __init__(self, shard_defs, mode="rb", fs=None, chunksize=1000):
      self.shard_defs = shard_defs
      self.mode = "rb"
      self.fs = fs
      self.chunksize = chunksize

    def __iter__(self):
      len_self = len(self)
      for start in range(0, len_self, self.chunksize):
          end = min(len_self, start+self.chunksize)
          for line in self[start:end]:
            yield line

    def __len__(self):
        return max(block[-1] for block in self.shard_defs)

    def __getitem__(self, keys):
        start, end = 0, 0
        if isinstance(keys, int):
          contiguous = False
        elif isinstance(keys, slice):
          contiguous = True
          start = 0 if keys.start is None else keys.start
          end = len(self) if keys.stop is None else keys.stop
        else:
          contiguous, start, end = is_contiguous(keys)
        if contiguous:
          if start >= len(self) or end > len(self):
            raise RuntimError(f"indexes {start}..{end} out of range")
          ret = []
          for block in self.shard_defs:
            file, start0, end0 = block['file'], block['start'], block['end']
            if (start <= end0 and end >= start0):
              if 'fobj' not in block or block['fobj'] is None:
                fobj = block['fobj'] = IndexGzipFileExt.open(file, mode=self.mode, fs=self.fs)
              ret.extend(fobj[slice(start-start0, min(min0, min)-start0)])
            elif end < start0 or end < end0:
              break
          return ret
        elif isinstance(keys, int):
          if keys >= len(self):
            raise RuntimError(f"index {keys} out of range")
          idx = keys
          for block in self.shard_defs:
            file, start0, end0 = block['file'], block['start'], block['end']
            if (start <= end0 and end >= start0):
              if 'fobj' not in block or block['fobj'] is None:
                fobj = block['fobj'] = IndexGzipFileExt.open(file, mode=self.mode, fs=self.fs)
              return fobj[indx-start0]
            elif end < start0 or end < end0:
              break
          return None
        else:
          return [self[idx] for idx in keys]


class IndexGzipFileExt(igzip.IndexedGzipFile):
    """This class inheriets from ``ingdex_gzip.IndexedGzipFile``.
    This class allows in addition to the functionality of
    IndexedGzipFile, access to a specific line based on the seek point
    of the line, using the __getitem__ method.  Additionally, a
    (conginguous) list or slice can be used, which will be more
    efficient then doing line by line access.
    
    The base IndexedGzipFile class allows for fast random access of a gzip
    file by using the ``zran`` library to build and maintain an index of seek
    points into the file.
    ``IndexedGzipFile`` is an ``io.BufferedReader`` which wraps an
    :class:`_IndexedGzipFile` instance. By accessing the ``_IndexedGzipFile``
    instance through an ``io.BufferedReader``, read performance is improved
    through buffering, and access to the I/O methods is made thread-safe.
    A :meth:`pread` method is also implemented, as it is not implemented by
    the ``io.BufferedReader``.
    """


    def __init__(self, *args, **kwargs):
        """Create an ``LineIndexGzipFile``. The file may be specified either
        with an open file handle (``fileobj``), or with a ``filename``. If the
        former, the file must have been opened in ``'rb'`` mode.

        .. note:: We do not support ``auto_build`` behaviour provided
        under IndexedGzipFile as the index is either built on load or
        saved directly in the .igz (pickle) file.

        :arg filename:         File name or open file handle.
        :arg fileobj:          Open file handle.
        :arg mode:             Opening mode. Must be either ``'r'`` or ``'rb``.
        :arg skip_crc_check:   Defaults to ``False``. If ``True``, CRC/size
                               validation of the uncompressed data is not
                               performed.
        :arg spacing:          Number of bytes between index seek points.
        :arg window_size:      Number of bytes of uncompressed data stored with
                               each seek point.
        :arg readbuf_size:     Size of buffer in bytes for storing compressed
                               data read in from the file.
        :arg readall_buf_size: Size of buffer in bytes used by :meth:`read`
                               when reading until EOF.
        :arg drop_handles:     Has no effect if an open ``fid`` is specified,
                               rather than a ``filename``.  If ``True`` (the
                               default), a handle to the file is opened and
                               closed on every access. Otherwise the file is
                               opened at ``__cinit__``, and kept open until
                               this ``_IndexedGzipFile`` is destroyed.
        :arg index_file:       Pre-generated index for this ``gz`` file -
                               if provided, passed through to
                               :meth:`import_index`.
        :arg buffer_size:      Optional, must be passed as a keyword argument.
                               Passed through to
                               ``io.BufferedReader.__init__``. If not provided,
                               a default value of 1048576 is used.
        :arg line2seekpoint:      Optional, must be passed as a keyword argument.
                               If not passed, this will automatically be created.    
        :arg file_size:      Optional, must be passed as a keyword argument.
                               If not passed, this will automatically be created.                             

        """
        self.line2seekpoint        = kwargs.pop('line2seekpoint', None)
        self.file_size        = kwargs.pop('file_size', None)
        kwargs['auto_build'] = False
        super().__init__(*args, **kwargs)
        if self.file_size is None:
          try:
            pos = self.tell()
            self.seek(0, os.SEEK_END)
            self.file_size = self.tell() 
            self.seek(pos, 0)
          except:
            self.build_full_index()
            pos = self.tell()
            self.seek(0, os.SEEK_END)
            self.file_size =  self.tell() 
            self.seek(pos, 0)

        if self.line2seekpoint is None:
          self.line2seekpoint=[]
          with self._IndexedGzipFile__file_lock:
            pos = self.tell()
            self.seek(0, 0)
            self.line2seekpoint.append(0)
            while True:
              line = self.readline().decode()
              if not line:
                break
              self.line2seekpoint.append(self.tell())
            self.seek(pos, 0)
    
    @classmethod
    def open(cls, f, mode="rb", fs=None, dataset_path=None):
      if not f.endwith(".gz") and not f.endswith(".igz"):
        raise RuntimeError("file of wrong format. must be .gz or .igz")
      if fs is None:
        fs = os
      if dataset_path is None:
        dataset_path = get_temporary_cache_files_directory()
      if is_remote_filesystem(fs):
        data_path = os.path.join(dataset_path, f)
        f_dataset_path = extract_path_from_uri(data_path) 
        tmp_dir = tempfile.TemporaryDirectory()
        data_path = Path(tmp_dir.name, f_dataset_path)
        fs.download(f_dataset_path, data_path.as_posix(), recursive=True)
        if f.endswith(".gz"):
          f = data_path
          f_dataset_path2 = f_dataset_path.replace(".gz", ".igz")
          data_path2 = Path(tmp_dir.name, f_dataset_path2)
          fs.download(f_dataset_path2, data_path2.as_posix(), recursive=True)
        elif f.endswith(".gz"):
          f_dataset_path2 = f_dataset_path.replace(".igz", ".gz")
          data_path2 = Path(tmp_dir.name, f_dataset_path2)
          f = data_path2
          fs.download(f_dataset_path2, data_path2.as_posix(), recursive=True)
      next(wait_until_files_loaded(f))
      if f.endswith(".igz"): 
        f = f.replace(".igz",".gz")
        next(wait_until_files_loaded(f))
      if not os.path.exists(f.replace(".gz",".igz")):
          fobj = cls(f, mode=mode)
          with open(f.replace(".gz",".igz"), "wb") as file:
            pickle.dump(fobj, file, pickle.HIGHEST_PROTOCOL)
      cwd = os.getcwd()
      dir = os.path.abspath(os.path.dirname(f))
      f = os.path.basename(f)
      if dir:
        os.chdir(dir)
      with open(f.replace(".gz",".igz"), "rb") as file:
        fobj = pickle.load(file)
      os.chdir(cwd)
      return fobj


    @staticmethod
    def unpickle(state):
      """Create a new ``IndexedGzipFile`` from a pickled state.
      :arg state: State of a pickled object, as returned by the
                  ``IndexedGzipFile.__reduce__`` method.
      :returns:   A new ``IndexedGzipFile`` object.
      """

      tell  = state.pop('tell')
      index = state.pop('index')
      state['filename'] = os.path.join(os.getcwd(), os.path.basename(state['filename']))
      gzobj = IndexGzipFileExt(**state)

      if index is not None:
          gzobj.import_index(fileobj=io.BytesIO(index))

      gzobj.seek(tell)

      return gzobj

    def __reduce__(self):
      """Used to pickle an ``LineIndexGzipFile``.
      Returns a tuple containing:
        - a reference to the ``unpickle`` function
        - a tuple containing a "state" object, which can be passed
          to ``unpickle``.
      """

      fobj = self._IndexedGzipFile__igz_fobj

      if (not fobj.drop_handles) or (not fobj.own_file):
          raise pickle.PicklingError(
              'Cannot pickle IndexedGzipFile that has been created '
              'with an open file object, or that has been created '
              'with drop_handles=False')

      # export and serialise the index if
      # any index points have been created.
      # The index data is serialised as a
      # bytes object.
      if fobj.npoints == 0:
          index = None

      else:
          index = io.BytesIO()
          self.export_index(fileobj=index)
          index = index.getvalue()

      state = {
          'filename'         : fobj.filename,
          'auto_build'       : fobj.auto_build,
          'spacing'          : fobj.spacing,
          'window_size'      : fobj.window_size,
          'readbuf_size'     : fobj.readbuf_size,
          'readall_buf_size' : fobj.readall_buf_size,
          'buffer_size'      : self._IndexedGzipFile__buffer_size,
          'line2seekpoint'   : self.line2seekpoint,
          'file_size'   : self.file_size,
          'tell'             : self.tell(),
          'index'            : index}

      return (IndexGzipFileExt.unpickle, (state, ))

    @property
    def filename(self):
      return self._IndexedGzipFile__igz_fobj.filename

    def __iter__(self):
      len_self = len(self)
      for start in range(0, len_self, 1000):
          end = min(len_self, start+1000)
          for line in self[start:end]:
            yield line

    def __len__(self):
      return len(self.line2seekpoint)

    def __getitem__(self, keys):
      start, end = 0, 0
      if isinstance(keys, int):
        contiguous = False
      elif isinstance(keys, slice):
        contiguous = True
        start = 0 if keys.start is None else keys.start
        end = len(self) if keys.stop is None else keys.stop
      else:
        contiguous, start, end = is_contiguous(keys)
      if contiguous:
        if start >= len(self) or end > len(self):
          raise RuntimError(f"indexes {start}..{end} out of range")
        start = self.line2seekpoint[start]
        if end == len(self):
          end = self.file_size
        else:
          end= self.line2seekpoint[end+1]-1
        with self._IndexedGzipFile__file_lock:
          pos = self.tell()
          self.seek(start, 0)
          ret= self.read(end-start).decode().split('\n')
          self.seek(pos, 0)
          return ret
      elif isinstance(keys, int):
        if keys >= len(self):
          raise RuntimError(f"index {keys} out of range")
        start = self.line2seekpoint[keys]
        with self._IndexedGzipFile__file_lock:
          pos = self.tell()
          self.seek(start, 0)
          ret= self.readline().decode()
          self.seek(pos, 0)
          return ret
      else:
        return [self[idx] for idx in keys]


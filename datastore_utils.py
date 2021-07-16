#%%writefile data-tooling/datastore_utils.py
#Copyright July 2021 Ontocord LLC. Licensed under Apache v2 https://www.apache.org/licenses/LICENSE-2.0


from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union
from typing import TYPE_CHECKING, Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from datasets.info import DatasetInfo
from datasets.features import PandasArrayExtensionArray, PandasArrayExtensionDtype, Features, Value, cast_to_python_objects, pandas_types_mapper
from datasets import utils, Dataset
from datasets.splits import NamedSplit
from datasets.arrow_writer import ArrowWriter, OptimizedTypedSequence
import os
import io
import json
from pathlib import Path
from datasets.utils.typing import PathLike
from datasets.arrow_dataset import transmit_format# , replayable_table_alteration
#from transformers import PreTrainedModel, PretrainedConfig
import copy
import shutil
import sqlalchemy 

from datasets.fingerprint import (
    fingerprint_transform,
    generate_fingerprint,
    generate_random_fingerprint,
    get_temporary_cache_files_directory,
    is_caching_enabled,
    update_fingerprint,
)
from datasets.dataset_dict import DatasetDict
from torch import nn
import pickle
import threading

import glob, shutil, os, time
import indexed_gzip as igzip
#import zstandard, io
#from gzip_stream import GZIPCompressedStream
import  fsspec.compression
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.schema import MetaData
from sqlalchemy.pool import StaticPool
from sqlalchemy.util import safe_reraise
from sqlalchemy.engine.reflection import Inspector
from dataset.types import Types

#from flask_sqlalchemy import SQLAlchemy
#from flask import Flask

#TODO: consider adding flask + gevent + oauth2 support for built in model-view-controllers and permissions for Datastores of various types.
#this will allow regional partners to permit searching and serving their own dataset and control access to them.

import dataset
from dataset.util import normalize_table_name
from dataset.util import DatasetException, ResultIter, QUERY_STEP
import six
from six.moves.urllib.parse import parse_qs, urlparse

import IPython
#import ipysheets
import pandas as pd
def display_results(batch):
    IPython.display(pd.DataFrame(batch))

# if an array is contiguous, return True, and the start and end+1 range usable in 'range(start, end)'
def is_contiguous(arr):
        start = None
        prev = None
        contiguous=True
        for i in arr:
          if start is None:
            start = i
          if prev is None or i == prev+1:
            prev = i
            continue
          contiguous = False
          break
        return contiguous, start, i+1

# This is used for seeing if files from gdrive from colab have finished loading. Files could be in flight while we try to retreive them.
def wait_until_files_loaded(flist, max_tries=120): # wait 2 hrs max
  ret_str =False
  if isinstance(flist, str):
    ret_str= True
    flist = [[flist, 0]]
  else:
    flist = [[f, 0]for f in flist]
  for j in range(len(flist)*max_tries):
    num_done = 0
    for i, val in enumerate(flist):
      if val is None:
        num_done += 1
        continue
      (f, incr) = val
      if incr > max_tries:
        raise RuntimeError("Timed out while trying to wait for file " + str(f))
      size1 = os.stat(f).st_size
      time.sleep(min(600, 1 + incr))
      incr += 1
      if os.stat(f).st_size == size1:
        flist[i] = None
        num_done += 1
        yield f
      else:
        flist[i]=[f,incr]
    if num_done == len(flist):
      return
  raise RuntimeError("Something went really wrong")

# just a wrapper to load igzip and regular .txt/.csv/.tsv files
def get_file_read_obj(f, mode="rb"):
  next(wait_until_files_loaded(f))
  if f.endswith(".gz"):
    if not os.path.exists(f.replace(".gz",".igz")):
        fobj = IndexGzipFileExt(f)
        fobj.build_full_index()
        with open(f.replace(".gz",".igz"), "wb") as file:
          pickle.dump(fobj, file, pickle.HIGHEST_PROTOCOL)
    else:
      cwd = os.getcwd()
      dir = os.path.abspath(os.path.dirname(f))
      f = os.path.basename(f)
      if dir:
        os.chdir(dir)
      with open(f.replace(".gz",".igz"), "rb") as file:
        fobj = pickle.load(file)
      os.chdir(cwd)
    return fobj
  else:
    return open(f, mode)


# getting file size, working with igzip files and regular txt files
def get_file_size(fobj):
  if not isinstance(fobj, IndexGzipFileExt):
    return os.stat(fobj).st_size
  else:  
    return fobj.file_size


# break up a file into shards, ending each shard on a line break
def get_file_segs_lines(input_file_path, file_seg_len=1000000, num_segs=None):
      f = get_file_read_obj(input_file_path)
      file_size= get_file_size(f)       
      file_segs = []
      if num_segs is not None:
          file_seg_len = int(file_size/num_segs)

      file_pos = 0
      while file_pos < file_size:
            if file_size - file_pos <= file_seg_len:
                file_segs.append((file_pos, file_size - file_pos))
                break
            f.seek(file_pos+file_seg_len, 0)
            seg_len = file_seg_len
            line = f.readline()
            if not line:
                file_segs.append((file_pos, file_size - file_pos))
                break
            seg_len += len(line)
            if file_size-(file_pos+seg_len) < file_seg_len:
                file_segs.append((file_pos, file_size - file_pos))
                break

            file_segs.append((file_pos, seg_len))
            file_pos = f.tell()
      f.close()
      line = None
      return file_segs

class TableExt(dataset.Table):
    """ Extends dataset.Table's functionality to work with Datastore(s) and to add full-text-search (FTS) """

    def find(self, *_clauses, **kwargs):
        """Perform a simple search on the table similar to
        dataset.Table's find, except: optionally gets a result only
        for specific columns by passing in _columns keyword.

        # TODO, full text search

        Simply pass keyword arguments as ``filter``.
        ::

            results = table.find(country='France')
            results = table.find(country='France', year=1980)

        Using ``_limit``::

            # just return the first 10 rows
            results = table.find(country='France', _limit=10)

        You can sort the results by single or multiple columns. Append a minus
        sign to the column name for descending order::

            # sort results by a column 'year'
            results = table.find(country='France', order_by='year')
            # return all rows sorted by multiple columns (descending by year)
            results = table.find(order_by=['country', '-year'])

        To perform complex queries with advanced filters or to perform
        aggregation, use :py:meth:`db.query() <dataset.Database.query>`
        instead.
        """

        if not self.exists:
            return iter([])

        _fts_q = kwargs.pop('_fts_q', None)
        _columns = kwargs.pop('_columns', None)
        _limit = kwargs.pop('_limit', None)
        _offset = kwargs.pop('_offset', 0)
        order_by = kwargs.pop('order_by', None)
        _streamed = kwargs.pop('_streamed', False)
        _step = kwargs.pop('_step', QUERY_STEP)
        if _step is False or _step == 0:
            _step = None
        if type(_columns) is str: _columns = [_columns]
        fts_results = []
        
        if _fts_q:
            fts_q = " AND ".join([f"{column} MATCH {q}" for column, q in _fts_q])
            # we could run a seperate sqlite database for fts and join manually using a list of id's
            # TODO: if we are doing fts against the same db then we want to combine in one query.  
            fts_results = self.fts_db.executable.execute(f"""SELECT id, rank
                              FROM {self.table_name}_idx
                              WHERE {fts_q}
                              ORDER BY rank
                              LIMIT {_limit}""").fetchall() # TODO: use parameters
            order_by = self._args_to_order_by(order_by) #TODO, if the order_by is empty, we will order by rank.
            args = self._args_to_clause(kwargs, clauses=_clauses) #todo, add in the where clause.  WHERE id in (...)
        else:
            order_by = self._args_to_order_by(order_by)
            args = self._args_to_clause(kwargs, clauses=_clauses)

        if _columns is None:
            query = self.table.select(whereclause=args,
                                  limit=_limit,
                                  offset=_offset)
        else:
            query = self.table.select(whereclause=args,
                                  limit=_limit,
                                  offset=_offset).with_only_columns([self.table.c[col] for col in _columns])

        if len(order_by):
            query = query.order_by(*order_by)
        
        conn = self.db.executable
        if _streamed:
            conn = self.db.engine.connect()
            conn = conn.execution_options(stream_results=True)
            
        return ResultIter(conn.execute(query),
                          row_type=self.db.row_type,
                          step=_step)


    def insert(self, row, ensure=None, types=None):
        """Add a ``row`` dict by inserting it into the table.

        If ``ensure`` is set, any of the keys of the row are not
        table columns, they will be created automatically.

        During column creation, ``types`` will be checked for a key
        matching the name of a column to be created, and the given
        SQLAlchemy column type will be used. Otherwise, the type is
        guessed from the row value, defaulting to a simple unicode
        field.
        ::

            data = dict(title='I am a banana!')
            table.insert(data)

        Returns the inserted row's primary key.
        """
        # either sqlalachemy or the underlying sqlite database starts auto-numbering at 1. we want to auto-number starting at 0
        # we shold probably check the count of the row, but this would require a round trip
        # to the db on each insert, so we'll make the assumption of lazy creation
        if (not self.exists or not self.has_column(self._primary_id)) and  self._primary_type in (Types.integer, Types.bigint):
          row[self._primary_id] = 0
        return super().insert(row,  ensure=ensure, types=types)

    def insert_many(self, rows, chunk_size=1000, ensure=None, types=None):
        """Add many rows at a time.

        This is significantly faster than adding them one by one. Per default
        the rows are processed in chunks of 1000 per commit, unless you specify
        a different ``chunk_size``.

        See :py:meth:`insert() <dataset.Table.insert>` for details on
        the other parameters.
        ::

            rows = [dict(name='Dolly')] * 10000
            table.insert_many(rows)
        """
        row = copy.copy(rows[0])
        rows[0] = row
        if (not self.exists or not self.has_column(self._primary_id)) and  self._primary_type in (Types.integer, Types.bigint) and self._primary_id not in row:
          row[self._primary_id] = 0
        return super().insert_many(rows, chunk_size=chunk_size, ensure=ensure, types=types)


  # TODO - do update with re-try

class DatabaseExt(dataset.Database):
    """A DatabaseExt textends dataset.Database and represents a SQL database with multiple tables of type TableExt."""

    def __init__(self, *args, **kwargs ):
        """Configure and connect to the database."""
        super().__init__(*args, **kwargs)


    # TODO: not currenlty working
    # will only work for sqlite. 
    # diferent databases have different fts. 
    def create_fts_index_column(self, table_name, column, stemmer="unicode61"): #  porter 
        # maybe we create a mirror sqlite database called fts_db if the database we are opening is not of sqlite type.
        # the idea is we want to be able to locally attach fts with our datasets arrow files. 
        self.db.executeable.execute(f'CREATE VIRTUAL TABLE {table_name}_idx USING FTS5(id:INTEGER, {column}:VARCHAR, tokenize="{stemmer}");')

    def create_table(
        self, table_name, primary_id=None, primary_type=None, primary_increment=None
    ):
        """Create a new table.

        Either loads a table or creates it if it doesn't exist yet. You can
        define the name and type of the primary key field, if a new table is to
        be created. The default is to create an auto-incrementing integer,
        ``id``. You can also set the primary key to be a string or big integer.
        The caller will be responsible for the uniqueness of ``primary_id`` if
        it is defined as a text type. You can disable auto-increment behaviour
        for numeric primary keys by setting `primary_increment` to `False`.

        Returns a :py:class:`Table <dataset.Table>` instance.
        ::

            table = db.create_table('population')

            # custom id and type
            table2 = db.create_table('population2', 'age')
            table3 = db.create_table('population3',
                                     primary_id='city',
                                     primary_type=db.types.text)
            # custom length of String
            table4 = db.create_table('population4',
                                     primary_id='city',
                                     primary_type=db.types.string(25))
            # no primary key
            table5 = db.create_table('population5',
                                     primary_id=False)
        """
        assert not isinstance(
            primary_type, str
        ), "Text-based primary_type support is dropped, use db.types."
        table_name = normalize_table_name(table_name)
        with self.lock:
            if table_name not in self._tables:
                self._tables[table_name] = TableExt(
                    self,
                    table_name,
                    primary_id=primary_id,
                    primary_type=primary_type,
                    primary_increment=primary_increment,
                    auto_create=True,
                )
            return self._tables.get(table_name)

    def load_table(self, table_name):
        """Load a table.

        This will fail if the tables does not already exist in the database. If
        the table exists, its columns will be reflected and are available on
        the :py:class:`Table <dataset.Table>` object.

        Returns a :py:class:`Table <dataset.Table>` instance.
        ::

            table = db.load_table('population')
        """
        table_name = normalize_table_name(table_name)
        with self.lock:
            if table_name not in self._tables:
                self._tables[table_name] = TableExt(self, table_name)
            return self._tables.get(table_name)



class IndexGzipFileExt(igzip.IndexedGzipFile):
    """This class inheriets from `` ingdex_gzip.IndexedGzipFile``. This class allows in addition to the functionality 
    of IndexedGzipFile, access to a specific line based on the seek point of the line, using the __getitem__ method.

    Additionally, a (conginguous) list or slice can be used, which will be more efficient then doing line by line access. 
    
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
        .. note:: The ``auto_build`` behaviour only takes place on calls to
                  :meth:`seek`.
        :arg filename:         File name or open file handle.
        :arg fileobj:          Open file handle.
        :arg mode:             Opening mode. Must be either ``'r'`` or ``'rb``.
        :arg auto_build:       If ``True`` (the default), the index is
                               automatically built on calls to :meth:`seek`.
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
              #print(line)
              if not line:
                break
              self.line2seekpoint.append(self.tell())

            self.seek(pos, 0)
          

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



if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    from datasets import load_dataset
    args = sys.argv[1:]
    if "-test_igzip" in args:
      if not os.path.exists("wikitext-2"):
        os.system('wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip')
        os.system('unzip wikitext-2-v1.zip')
        os.system('gzip wikitext-2/wiki.train.tokens')
      vi1 = get_file_read_obj("wikitext-2/wiki.train.tokens.gz")
      assert len(vi1[[0, 5, 1000]]) == 3
      assert len(vi1[0:]) == len(vi1)
    if "-test_sql" in args:
      db = DatabaseExt("sqlite://")
      table = db['user']
      assert table.exists == False
      assert table.has_column('id') == False
      assert table.insert(dict(name='John Doe', age=37)) == 0
      assert table.exists == True
      assert table.has_column('id') == True
      assert table.insert(dict(name='Jane Doe', age=20)) == 1
      jane  = table.find_one(name='Jane Doe')
      assert jane['id'] == 1

      db = DatabaseExt("sqlite://")
      table = db['user']
      rows = [dict(name='Dolly')] * 10
      table.insert_many(rows)
      assert list(table.find(id={'in':range(0, 2)}))[-1]['id'] == 1

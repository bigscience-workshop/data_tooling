#Copyright 2021, Ontocord, LLC
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

""" Utilities to access dataset/SqlAlchemy's sql utilities"""


from dataclasses import asdict
from collections.abc import Iterable
from collections import OrderedDict
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
import json
from pathlib import Path
from pathlib import PurePath
from datasets.utils.typing import PathLike
from datasets.arrow_dataset import map_function, transmit_format# , replayable_table_alteration

import copy
import shutil
from datasets.fingerprint import (
	fingerprint_transform,
	generate_fingerprint,
	generate_random_fingerprint,
	get_temporary_cache_files_directory,
	is_caching_enabled,
	update_fingerprint,
	)

from datasets.search import BaseIndex, BatchedSearchResults, SearchResults
from datasets.tasks import TaskTemplate
from datasets.table import InMemoryTable,  concat_tables
from datasets.dataset_dict import DatasetDict
from datasets import config
from datasets.filesystems import extract_path_from_uri, is_remote_filesystem
from datasets.utils import logging, map_nested
        
from torch import nn
import pickle
import glob, shutil, os, time
import indexed_gzip as igzip
import zipfile
import  fsspec.compression

import dataset
import six
from six.moves.urllib.parse import parse_qs, urlparse
import threading

from sqlalchemy.exc import ResourceClosedError
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.schema import MetaData
from sqlalchemy.pool import StaticPool
from sqlalchemy.util import safe_reraise
from sqlalchemy.engine.reflection import Inspector
from dataset.types import Types
from dataset.util import DatasetException, ResultIter, QUERY_STEP, row_type, normalize_table_name, convert_row

import dask
import dask.array as da
from dask.distributed import Client

from getpass import getpass
import atexit, os, subprocess
import requests
import atexit
import uuid
import multiprocessing
from smart_open import open
import urllib


import random
import socket
import copy
import itertools
from datetime import datetime, timedelta
import signal
import atexit
import warnings

from pandas import DataFrame, read_csv
import platform
import subprocess
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
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

from datastore.utils.persisted_row_shards import *

  
######################################################################################
# SqlAlchemy/Dataset based classes for interconnect with SQL, and in particular sqlite3
### A note about naming: dataset and datasets are not the same libraries.

def multi_iter_result_proxy(rps, step=None, batch_fns=None, row_types=row_type):
    """Iterate over the ResultProxyExt."""
    for batch_fn, row_type, rp in zip(batch_fns, row_types, rps):
      if step is None:
        chunk = rp.fetchall()
      else:
        chunk = rp.fetchmany(size=step)
      if batch_fn is not None:
        chunk = batch_fn(chunk)
      if not chunk:
        next
      if row_type is None:
        for row in chunk:
          yield row
      else:
        for row in chunk:
          yield convert_row(row_type, row)


class ResultIterExt():
    """Similar to dataset.ResultIter except thre are several
    result_proxies, and they are actually results that are lazy
    executed. Optionally, the results can be treated as a sql table via .to_sql()
    """

    def __init__(self, result_proxy, step=None, row_type=row_type, batch_fn=None):
      self.persisted=None
      if type(result_proxy) is not list:
          result_proxy = [result_proxy]
      self.result_proxy = result_proxy
      if type(row_type) is not list:
        self.row_type = [row_type]*len(self.result_proxy)
      else:
        self.row_type = row_type
      if type(batch_fn) is not list:
        self.batch_fn = [batch_fn]*len(self.result_proxy)
      else:
        self.batch_fn = batch_fn
      self.step = step
      try:
          self.keys = list(self.result_proxy[0].keys())
          #self.keys.append('rank')
          #print (self.keys)
          self._iter = multi_iter_result_proxy(self.result_proxy, step=self.step, batch_fns=self.batch_fn, row_types=self.row_type)
      except ResourceClosedError:
          self.keys = []
          self._iter = iter([])


    def to_sql(self, primary_id="id", primary_type=None, primary_increment=None):
      if self.persisted is not None:
        return self.persisted
      self.persisted  = TableExt(DatabaseExt("sqlite://"), str(self), primary_id=primary_id, primary_type=primary_type, primary_increment=primary_increment, auto_create=True)      
      self.persisted.insert_many(self)
      self._iter = multi_iter_result_proxy(self.persisted.db.executable.execute(self.persisted.table.select()), row_types=self.row_type)
      return self.persisted

    #extending while iterating will cause undefined behaviour
    def extend(self, result_iter):
      self.result_proxy.extend(result_iter.result_proxy)
      self.batch_fn.extend(result_iter.batch_fn)
      self.row_type.extend(result_iter.row_type)
      if self.persisted:
        self.persisted.insert_many(result_iter)
        self._iter = multi_iter_result_proxy(self.persisted.db.executable.execute(self.persisted.table.select()), row_types=self.row_type)
      else:
        try:
          self._iter = multi_iter_result_proxy(self.result_proxy, step=self.step, batch_fns=self.batch_fn, row_types=self.row_type)
        except ResourceClosedError:
          self.keys = []
          self._iter = iter([])

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self.close()
            raise

    next = __next__

    def __iter__(self):
        return self

    def close(self):
        for rp in self.result_proxy:
           rp.close()

class TableSharded(dataset.Table, PersistedRowShards):
    PRIMARY_DEFAULT = "id"

    """ Extends dataset.Table's functionality to work with
    Datastore(s) and to add sqlite full-text-search (FTS). Can be a
    view into several sharded tables in one or more databases. Sharded
    by the row primary id.

    We can treat sqlite databases in a special way because they are
    file based, and we lazy load those from a network or shared fs,
    etc.
    """

    def __init__(
        self,
        database=None,
        table_name=None,
        primary_id=None,
        primary_type=None,
        primary_increment=None,
        auto_create=False,
        shard_defs=None,
        start_row=None,
        end_row=None,
        fs = None,
    ):
      assert shards is not None or database is not None
      self.auto_create = auto_create
      self.start_row=start_row
      self.start_row=end_row

      auto_create = False
      super(dataset.Table).__init__(database, table_name, primary_id=primary_id, primary_type=primary_type, primary_increment=primary_increment, auto_create=auto_create)
      super(PersistedRowShards).__init(shard_defs, fs)
      # TODO: automatically fill in external_fts_columns and has_fts_trigger in the _sync* methods.
      self.external_fts_columns = {}
      self.has_fts_trigger = False

    def _shard_by_idx(self, idx):
      shard_def = self.shard_defs[idx]
      if shard_def['database_kwargs'] and 'url' in shard_def['database_kwargs']:
        url = shard_def['database_kwargs']['url']
        if url.startswith("sqlite:///"):
          new_path = "sqlite:///"+self.cache_shard_file(idx, url.replace("sqlite:///", ""))
          shard_def = copy.deepcopy(shard_def)
          shard_def['database_kwargs']['url'] = url
      elif shard_def['database_args']:
        url = shard_def['database_args']
        if url.startswith("sqlite:///"):
          new_path = "sqlite:///"+self.cache_shard_file(idx, url.replace("sqlite:///", ""))
          shard_def = copy.deepcopy(shard_def)
          shard_def['database_args'] = [url]
      return TableSharded(database=DatabaseExt(*shard_def['database_args'], **shard_def['database_kwargs']), table_name = self.name, primary_id= self._primary_id, primary_type= self._primary_type, primary_increment=self._primary_increment, auto_create=self.auto_create,  start_row=shard_def['start_row'], end_row=shard_def['end_row'],)

    def _sync_all_shards(self):
      if self.shards:
        for idx, shard_def, shard in enumerate(self.shard_defs):
          self.shards(idx)

      
    @property
    def exists(self):
        """Check to see if the table currently exists in the database."""
        if self.shards:
          self._sync_all_shards()
          for shard in self.shards:
            if shard.exits: return True
          return False
        else:
          return super().exits


    @property
    def table(self):
        """Get a reference to the table, which may be reflected or created."""
        if self.shards_defs:
          return self.shards(0).table
        else:
          return super().table

    @property
    def columns(self):
        """Get a listing of all columns that exist in the table."""
        if self.shards_defs:
          return self.shards(0).columns
        else:
          return super().columns

    def has_column(self, column):
        """Check if a column with the given name exists on this table."""
        if self.shards_defs:
          return self.shards(0).has_column(column)
        else:
          return super().has_column(column)

    def create_column(self, name, type, **kwargs):
        """Create a new column ``name`` of a specified type.
        ::

            table.create_column('created_at', db.types.datetime)

        `type` corresponds to an SQLAlchemy type as described by
        `dataset.db.Types`. Additional keyword arguments are passed
        to the constructor of `Column`, so that default values, and
        options like `nullable` and `unique` can be set.
        ::

            table.create_column('key', unique=True, nullable=False)
            table.create_column('food', default='banana')
        """
        if self.shards_defs:
          for idx in range(len(self.shards_defs)):
            shard = self.shards(idx)
            shard.create_column(name, type, **kwargs)
        else:
          super().create_column(name, type, **kwargs)


    def create_column_by_example(self, name, value):
        """
        Explicitly create a new column ``name`` with a type that is appropriate
        to store the given example ``value``.  The type is guessed in the same
        way as for the insert method with ``ensure=True``.
        ::

            table.create_column_by_example('length', 4.2)

        If a column of the same name already exists, no action is taken, even
        if it is not of the type we would have created.
        """
        if self.shards_defs:
          for idx in range(len(self.shards_defs)):
            shard = self.shards(idx)
            shard.create_column_by_example(name, value)
        else:
          super().create_column_by_example(name, value)


    def drop_column(self, name):
        """
        Drop the column ``name``.
        ::

            table.drop_column('created_at')

        """
        if self.shards_defs:
          for idx in range(len(self.shards_defs)):
            shard = self.shards(idx)
            shard.drop_column(column)
        else:
          super().drop_column(column)

    def drop(self):
        """Drop the table from the database.

        Deletes both the schema and all the contents within it.
        """
        if self.shards_defs:
          for idx in range(len(self.shards_defs)):
            shard = self.shards(idx)
            shard.drop()
        else:
          super().drop()

    def has_index(self, columns):
        """Check if an index exists to cover the given ``columns``."""
        if self.shards_defs:
          for idx in range(len(self.shards_defs)):
            shard = self.shards(idx)
            if shard.has_index(columns):
              return True
          return False
        else:
          return super().has_index(columns)

    def create_index(self, columns, name=None, **kw):
        """Create an index to speed up queries on a table.

        If no ``name`` is given a random name is created.
        ::

            table.create_index(['name', 'country'])
        """
        if self.shards_defs:
          for idx in range(len(self.shards_defs)):
            shard = self.shards(idx)
            shard.create_index(columns, name, **kw)
        else:
          super().create_index(columns, name, **kw)

    def find(self, *_clauses, **kwargs):
        """Perform a search on the table similar to
        dataset.Table.find, except: optionally gets a result only for
        specific columns by passing in _columns keyword. And
        optionally perform full-text-search (BM25) on via a Sqlite3
        table.

        :arg _fts_query:
        :arg _fts_step:
        :arg _columns:

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

        if self.shards:
          self._sync_all_shards()
          if '_count' in kwargs:
            return sum([shard for shard in sellf.find(*copy.deepcopy(_clauses), **copy.deepcopy(kwargs))])
          if self._primary_id in kwargs:
            # do the case where we have specific id ranges. we pass in the queries by row id.
            ids = kwars[self._primary_id]['in']
            ids.sort()
            ret = None
            shards_defs = self.shards_defs
            shard_idx = 0
            curr_ids = []
            for idx, _id in enumerate(ids):
              if not shard_defs: break
              if len(shards_defs) == 1:
                curr_ids = ids[idx:]
                break
              if _id >=  shards_defs[0]['start_row']  and _id <= shards_defs[0]['end_row']:
                curr_ids.append(_id)
              elif _id > shards_defs[0]['end_row']:
                if curr_ids:
                  kwargs_shard = copy.deepcopy(kwargs)
                  kwargs_shard[self._primary_id]={'in': curr_ids}
                  if not ret:
                    ret = self.shards(shard_idx).find(*copy.deepcopy(_clauses), **kwargs_shard)
                  else:
                    ret.extend(self.shards(shard_idx).find(*copy.deepcopy(_clauses), **kwargs_shard))
                shards_defs = shards_defs[1:]
                shard_idx += 1
                curr_ids = [_id]
            if curr_ids:
              kwargs_shard = copy.deepcopy(kwargs)
              kwargs_shard[self._primary_id]={'in': curr_ids}
              if not ret:
                ret = self.shards(shard_idx).find(*copy.deepcopy(_clauses), **kwargs_shard)
              else:
                ret.extend(self.shards(shard_idx).find(*copy.deepcopy(_clauses), **kwargs_shard))
            return ret
          else:
            ret = []
            for idx in range(len(self.shards_defs)):
              shard = self.shards(idx)
              if not ret:
                ret = shard.find(*copy.deepcopy(_clauses), **copy.deepcopy(kwargs))
              else:
                ret.extend(shard.find(*copy.deepcopy(_clauses), **copy.deepcopy(kwargs)))
            return ret
        
        _fts_query = kwargs.pop('_fts_query', None)
        _count = kwargs.pop('_count', None)
        _fts_step = kwargs.pop('_fts_step', QUERY_STEP)
        _columns = kwargs.pop('_columns', None)
        _distinct = kwargs.pop('_distinct', None)
        _limit = kwargs.pop('_limit', None)
        _offset = kwargs.pop('_offset', 0)
        order_by = kwargs.pop('order_by', None)
        _streamed = kwargs.pop('_streamed', False)
        _step = kwargs.pop('_step', QUERY_STEP)

        assert _count is None or _columns is None, "can only get the count when not selecting by columns"

        if _fts_query and _step < _fts_step:
          _step = _fts_step
        if _step is False or _step == 0:
            _step = None
        if type(_columns) is str: _columns = [_columns]
        fts_results = []
        if _fts_query is None:
          _fts_query=[]
        id2rank={}
        return_id2rank_only = False
        for column, q in _fts_query:
            if column in self.external_fts_columns:
              db, fts_table_name = self.external_fts_columns[column]
            else:
              # if the table being searched is an fts_idx (ends with
              # _fts_idx), assume there are other tables for each of
              # the columns being search in the format of
              # <name>_colummn_fts_idx
              if self.name.endswith(f"_fts_idx"):
                if self.name.endswith(f"_{column}_fts_idx"):
                  db, fts_table_name = self.db, self.name
                else:
                  db, fts_table_name = self.db, "_".join(self.name.replace("_fts_idx", "").split("_")[:-1])+f"_{column}_fts_idx"
                return_id2rank_only = True
              else:
                db, fts_table_name = self.db, f"{self.name}_{column}_fts_idx"
              # TODO: check if sqlite3 
            q = q.replace("'","\'") 
            if _limit is not None:
              # using _limit is a hack. we want a big enough result
              # from fts so that subsequent query will narrow down to a
              # good result at _limit, but we don't want all results
              # which could be huge. we can set the fts_max_limit as a
              # parameter to the table or db level.
              if  kwargs:
                new_limit = _limit*10 
              else:
                new_limit = _limit
            else:
                new_limit = 1000000
            args2 = ""
            if self._primary_id in kwargs:
                kwargs2={'rowid': kwargs[self._primary_id]}
                args2 = self._args_to_clause(kwargs2).replace("WHERE", "AND")
                    
            fts_results = db.executable.execute(f"""SELECT rank, rowid 
                                FROM {fts_table_name}
                                WHERE {column} MATCH '{q}' {args2}
                                ORDER BY rank
                                LIMIT {new_limit}""").fetchall() 
            if not fts_results:
                return []
            for a in fts_results:
              id2rank[a[1]] = min(a[0], id2rank.get(a[1], 1000))

        #we do an optimization here by just returning the id and/or rank, if there are no queries except for the id and/or rank
        if id2rank and ((_columns and ((self._primary_id,) == tuple(columns)) or ((self._primary_id, 'rank') == tuple( _columns))) or return_id2rank_only):
          ret = list(id2rank.items())
          ret.sort(key=lambda a: a[0])
          if _columns and len(_columns) == 1:
            return [a[1] for a in ret]
          else:
            return [{self._primary_id: a[1], 'rank': a[0]} for a in ret]
        if return_id2rank_only:
          return []
        order_by = self._args_to_order_by(order_by)
        conn = self.db.executable

        if _streamed:
            conn = self.db.engine.connect()
            conn = conn.execution_options(stream_results=True)
        
        # if the fts_results are quite large let's issue several queries by fts_step, and return an iterator with a lazy execution. 
        batch_fn = None
        results = []
        row_type_fn = self.db.row_type
        if id2rank:
          fts_results = list(id2rank.items())
          fts_results.sort(key=lambda a: a[0])          
          len_fts_results = len(fts_results)
          total_cnt = 0
          for rng in range(0, len_fts_results, _fts_step):
            max_rng = min(len_fts_results, rng+_fts_step)
            kwargs[self._primary_id] ={'in': [int(a[1]) for a in fts_results[rng:max_rng]]}
            args = self._args_to_clause(kwargs, clauses=_clauses)

            if _columns is None:
              if _count:
                query = select([func.count()], whereclause=args)
                query = query.select_from(self.table, 
                                          limit=_limit,
                                          offset=_offset)
              else:
                query = self.table.select(whereclause=args,
                                      limit=_limit,
                                      offset=_offset)
            else:
              query = self.table.select(whereclause=args,
                                      limit=_limit,
                                      offset=_offset).with_only_columns([self.table.c[col] for col in _columns])
            if len(order_by):
                query = query.order_by(*order_by)
            if _count:
              total_cnt += conn.execute(query).fetchone()[0]
            else:
              results.append(conn.execute(query))

          if len(order_by):
            def row_type_fn(row):
              row =  convert_row(self.db.row_type, row)
              row['rank'] = id2rank[row[self._primary_id]]
              return row
          else:
            # we will need to reorder the results based on the rank of the fts_results.
            def batch_fn(batch):
              batch2 = []
              for row in batch:
                row = convert_row(self.db.row_type, row)
                row['rank'] = id2rank[row[self._primary_id]]
                batch2.append(row)
              batch2.sort(key=lambda a: a['rank'])
              return batch2
            row_type_fn = None
          if _count:
            return total_cnt 
        else:
          args = self._args_to_clause(kwargs, clauses=_clauses)
          if _columns is None:
              query = self.table.select(whereclause=args,
                                        distinct=_distinct,
                                        limit=_limit,
                                        offset=_offset)
          else:
              query = self.table.select(whereclause=args,
                                        distinct=_distinct,
                                        limit=_limit,
                                        offset=_offset).with_only_columns([self.table.c[col] for col in _columns])

          if len(order_by):
              query = query.order_by(*order_by)
          if _count:
            return conn.execute(query).fetchone()[0]
          results=[conn.execute(query)]
        return ResultIterExt(results,
                          row_type=row_type_fn,
                          step=_step, batch_fn=batch_fn)


    def count(self, *_clauses, **kwargs):
        """Return the count of results for the given filter set."""
        if not self.exists:
            return 0
        kwargs['_count'] = True
        return self.find( *_clauses, **kwargs)


    def __len__(self):
        """Return the number of rows in the table."""
        return self.count()

    def distinct(self, *args, **_filter):
        if not self.exists:
            return iter([])
        columns = []
        clauses = []
        for column in args:
            if isinstance(column, ClauseElement):
                clauses.append(column)
            else:
                if not self.has_column(column):
                    raise DatasetException("No such column: %s" % column)
                columns.append(column)
        if not len(columns):
            return iter([])
        _filter['_columns'] = columns
        _filter['_distinct'] = True
        return self.filter(*clauses, **_filter)

    def apply_to_shards(self, rows, fn, fn_kwargs):
        rows.sort()
        ret = None
        shards = self.shards
        curr_rows = []
        for idx, _id in enumerate(rows):
          if len(shards) == 1:
            curr_rows = rows[idx:]
            break
          if _id >= shard[0].start_row  and _id <= shard[0].end_row:
            curr_rows.append(_id)
          elif _id > shard[0].end_row:
            if curr_rows:
              kwargs_shard = copy.deepcopy(kwargs)
              kwargs_shard[self._primary_id]={'in': curr_rows}
              if not ret:
                ret = shard.find(*copy.deepcopy(_clauses), **kwargs_shard)
              else:
                ret.extend(shard.find(*copy.deepcopy(_clauses), **kwargs_shard))
            shards = shards[1:]
            curr_rows = [_id]
        if curr_rows:
          kwargs_shard = copy.deepcopy(kwargs)
          kwargs_shard[self._primary_id]={'in': curr_rows}
          if not ret:
            ret = shard.find(*copy.deepcopy(_clauses), **kwargs_shard)
          else:
            ret.extend(shard.find(*copy.deepcopy(_clauses), **kwargs_shard))
          return ret
        else:
          ret = []
          for shard in self.shards:
            if not ret:
              ret = shard.find(*copy.deepcopy(_clauses), **copy.deepcopy(kwargs))
            else:
              ret.extend(shard.find(*copy.deepcopy(_clauses), **copy.deepcopy(kwargs)))
          return ret


    def update(self, row, keys, ensure=None, types=None, return_count=False):
      if self.shards:
          self._sync_all_shards()
      row = self._sync_columns(row, ensure, types=types)
      if [_ for column in row.keys() if column in self.external_fts_columns]:
        args, row = self._keys_to_args(row, keys)
        old = list(self.find(**args))
      else:
        old = None
      ret =  super().update(row, keys=keys, ensure=ensure, types=types, return_count=return_count)
      if old:
        for key in row.keys():
          self.update_fts(column=key, old_data = old, new_data=row, mode='update')
      return ret

    def update_many(self, rows, keys, chunk_size=1000, ensure=None, types=None):
      rows = self._sync_columns(rows, ensure=ensure, types=types)
      if [_ for column in rows[0].keys() if column in self.external_fts_columns]:
        args, row = self._keys_to_args(rows, keys) # this probably won't work
        old = list(self.find(**args))
      else:
        old = None
      ret =  super().update_many(rows, keys=keys, chunk_size=chunk_size, ensure=ensure, types=types)
      if old:
        for key in rows[0].keys():
          self.update_fts(column=key, old_data = old, new_data=rows, mode='update')
      return ret

        
    def delete(self, *clauses, **filters):
      if self.external_fts_columns:
        old = list(self.find(*clauses, **filters))
        if old:
          for key in old[0].keys():
            self.update_fts(column=key, old_data = old, mode='delete')
      return super().delete(*clauses, **filters)


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
        ret =  super().insert(row,  ensure=ensure, types=types)
        for key in row.keys():
          self.update_fts(column=key, new_data=row, mode='insert')
        return ret

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
        ret =  super().insert_many(rows, chunk_size=chunk_size, ensure=ensure, types=types)
        for key in row.keys():
          self.update_fts(column=key, new_data=rows, mode='insert')
        return ret

    # create a sqlite fts index for a column in this table
    def create_fts_index_column(self, column, stemmer="unicode61", db=None): #  porter 
      if db is None: db = self.db
      if not db.is_sqlite:
        raise RuntimeError("the db for the fts index must be sqlite")
      table_name= self.name
      if not self.has_column(column):
        self.create_column_by_example(column,'**')
      fts_table_name = f"{table_name}_{column}_fts_idx"
      db.create_fts_index(fts_table_name, stemmer=stemmer, column=column, table_name = table_name)
      self.external_fts_columns[column] = (db, fts_table_name)        

    # sqlite3 fts manual updates. we create TableSharded level updates when we don't have actual triggers in the sqlite3 database.
    def update_fts(self, column, new_data=None, old_data=None, mode='insert'):
      if column != self._primary_id and column in self.external_fts_columns:
        db, fts_table_name = self.external_fts_columns[column]
        db.update_fts(fts_table_name, column=column, new_data=new_data, old_data=old_data, mode=mode, primary_id=self._primary_id)
      

class DatabaseExt(dataset.Database):
    """A DatabaseExt textends dataset.Database and represents a SQL database with multiple tables of type TableSharded."""
    
    def __init__(self, *args, **kwargs ):
        """Configure and connect to the database."""
        super().__init__(*args, **kwargs)

    def create_fts_index(self, fts_table_name, stemmer, column, table_name=None, primary_id="id"):
      if not self.is_sqlite:
        raise RuntimeError("cannot create sqlite fts index in non sqlite table")
      if fts_table_name not in self.tables:
          if table_name:
            self.executable.execute(f"CREATE VIRTUAL TABLE {fts_table_name} USING fts5({column}, tokenize='{stemmer}', content='{table_name}', content_rowid='{primary_id}');")
            table = self.create_table(fts_table_name, primary_id="rowid")
            table.has_fts_trigger = True
            self.executable.execute(f"CREATE TRIGGER {table_name}_{column}_ai AFTER INSERT ON {table_name} BEGIN INSERT INTO {fts_table_name} (rowid, {column}) VALUES (new.{primary_id}, new.{column}); END;")
            self.executable.execute(f"CREATE TRIGGER {table_name}_{column}_ad AFTER DELETE ON {table_name} BEGIN INSERT INTO {fts_table_name} ({fts_table_name}, rowid, {column}) VALUES('delete', old.{primary_id}, old.{column}); END;")
            self.executable.execute(f"CREATE TRIGGER {table_name}_{column}_au AFTER UPDATE OF {column} ON {table_name} BEGIN INSERT INTO {fts_table_name} ({fts_table_name}, rowid, {column}) VALUES('delete', old.{primary_id}, old.{column}); INSERT INTO {table_name}_{column}_fts_idx(rowid, {column}) VALUES (new.{primary_id}, new.{column}); END;")
          else:
            self.executable.execute(f"CREATE VIRTUAL TABLE {fts_table_name} USING fts5({column}, content='');")
            table = self.create_table(fts_table_name, primary_id="rowid")
            table.has_fts_trigger = False

      else:
          # do we want to warn?
          print (f"warning: {fts_table_name} already in database")
          return

    # update the sqlite3 fts index
    def update_fts(self, fts_table_name, column, new_data=None, old_data=None, mode='insert', primary_id="id"):
      if not self.is_sqlite:
        raise RuntimeError("applying an fts update to a db that is not sqlite")
      if old_data is None and mode != 'insert':
        raise RuntimeError(f"need to provide old data in order to update or delete the fts")
      if fts_table_name not in self.tables:
        raise RuntimeError(f"there is no fts index column for {column}")
      fts_table = self[fts_table_name]
      if not hasattr(fts_table, 'has_fts_trigger') or not fts_table.has_fts_trigger:
        return
      if (new_data and type(new_data) is not list) or (old_data and type(old_data) is not list):
        if mode in ('update', 'delete'):
          fts_table.insert({fts_table_name: 'delete', 'rowid': old_data[primary_id], column: old_data[column]})
        if mode in ('update', 'insert'):
          fts_table.insert({'rowid': new_data[primary_id], column: new_data[column]})
      else:
        if mode in ('update', 'delete'):
          old_data = copy.deepcopy(old_data)
          for data in old_data:
            data['rowid']= data[primary_id]
            data[fts_table_name] = 'delete'
            del data[primary_id]
            for key in list(data.keys()):
              if key not in (fts_table_name, 'rowid', column):
                del data[key]
          fts_table.insert_many(old_data)
        if mode in ('update', 'insert'):
          new_data = copy.deepcopy(new_data)
          for data in new_data:
            data['rowid']= data[primary_id]
            del data[primary_id]
            for key in list(data.keys()):
              if key not in ('rowid', column):
                del data[key]
            fts_table.insert_many(new_data)
  
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
                self._tables[table_name] = TableSharded(
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
                self._tables[table_name] = TableSharded(self, table_name)
            return self._tables.get(table_name)


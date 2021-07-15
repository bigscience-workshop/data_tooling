#Copyright July 2021 Ontocord LLC. Licensed under Apache v2 https://www.apache.org/licenses/LICENSE-2.0
#%%writefile data-tooling/datastore.py

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
import json
from pathlib import Path
from datasets.utils.typing import PathLike
from datasets.arrow_dataset import transmit_format# , replayable_table_alteration
#from transformers import PreTrainedModel, PretrainedConfig
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
from datasets.table import InMemoryTable,  concat_tables
from datasets.dataset_dict import DatasetDict
from torch import nn
import pickle

import glob, shutil, os, time
import indexed_gzip as igzip
#import zstandard, io
#from gzip_stream import GZIPCompressedStream
import  fsspec.compression

#from flask_sqlalchemy import SQLAlchemy
#from flask import Flask
import dataset
import six
from six.moves.urllib.parse import parse_qs, urlparse
try:
  from datastore_utils import *
except:
  pass

### NOTE: dataset is a different package than datasets. We are using both packages.
### We want to have mutliple types of storage that ideally can be
### transported as a file transfer with an arrow dataset (perhaps a tar file?). So if we
### have <signature>.arrow, we may have fts_<signature>.db (for full
### text indexing sqlite database) and <signature>.db (sqlite database), and
### <siganture>.mmap (mmap file reprsenting a tensor), and
### <singature>.igz (if we wish to store some portion of the text
### columns in igzip format for compression and legacy purposes.


### A note about naming: datasets uses the terms features and columns interchangably.

def np_mmap(mmap_path, dtype, shape):
  if os.path.exists(mmap_path):
    return np.memmap(filename=mmap_path, mode="r+", dtype=np.dtype(dtype), shape=tuple(shape))
  else:
    return np.memmap(filename=mmap_path, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape))
      
class FeaturesWithViews(Features):
    def copy(self):
        ret= FeaturesWithViews(super().copy())
        if hasattr(self, "features_map"):
            ret.features_map = copy.deepcopy(self.features_map)
        return ret

    def __repr__(self):
        ret =  "{"+"\n\t\t".join([f"'{a[0]}': {a[1]}" for a in self.items() if a[0] not in self.features_map])
        if self.features_map:
            ret = ret+"\n\t\t"+"\n\t\t".join(f"'{a[0]}': View({a[1]})" for a in  self.features_map.items())
        ret +="\n}"
        return ret

class MetadataMixin:

    @classmethod
    def rowid_key_name():
        raise NotImplementedError()

    @classmethod
    def extract_rowid(batch):
        raise NotImplementedError()

    @classmethod
    def section_key_name():
        raise NotImplementedError()

    @classmethod
    def extract_section_labels(batch):
        raise NotImplementedError()

    @classmethod
    def url_key_name():
        raise NotImplementedError()
        
    @classmethod
    def extract_url(batch):
        raise NotImplementedError()

    @classmethod
    def outlinks_key_name():
        raise NotImplementedError()
                
    @classmethod
    def extract_outlinks(batch):
        raise NotImplementedError()

    @classmethod
    def publication_date_key_name():
        raise NotImplementedError()
                
    @classmethod
    def extract_publication_date(batch):
        raise NotImplementedError()

    @classmethod
    def lastupdate_date_key_name():
        raise NotImplementedError()
                
    @classmethod
    def extract_lastupdate_date(batch):
        raise NotImplementedError()

    @classmethod
    def uuid_hash_key_name():
        raise NotImplementedError()
                
    @classmethod
    def create_uuid_hash(batch):
        raise NotImplementedError()

    @classmethod
    def main_langid_key_name(batch):
        raise NotImplementedError()

    @classmethod
    def extract_main_langid(batch):
        raise NotImplementedError()

    @classmethod
    def other_langid_key_name(batch):
        raise NotImplementedError()

    @classmethod
    def extract_other_langid(batch):
        raise NotImplementedError()

    @classmethod
    def pii_multiple_key_names():
      raise NotImplementedError()

    @classmethod
    def detect_and_process_pii(batch):
        raise NotImplementedError()

class Datastore(Dataset): 
    """
    A class that wraps a Huggingface arrow based Dataset to provide some optimized reading and *writing* in various persistance backends. 
    Currently provides support for features bound to memmap, indexed gzip (igzip) file, and sqlalchemy databases.
    """
        
    def __repr__(self):
        ret = FeaturesWithViews(self._info.features)
        ret.features_map = {} if not hasattr(self, "features_map") else self.features_map
        return f"Datastore({{\n    features: {ret},\n    num_rows: {self.num_rows}\n}})"
    

    @classmethod
    def from_dataset(cls, dataset, features_map=None, shared_dir=None):
        self = cls(
            arrow_table=dataset._data,
            indices_table=dataset._indices,
            info=dataset.info.copy(),
            split=dataset.split,
            fingerprint=dataset._fingerprint,
        )

        if  hasattr(dataset, "mmap_access_cnt"):
          self.mmap_access_cnt=dataset.mmap_access_cnt
        else:
          self.mmap_access_cnt=0
        if  hasattr(dataset, "features_map"):
          self.features_map=copy.deepcopy(dataset.features_map)
        if features_map is not None:
          self.features_map = copy.deepcopy(features_map)
        if not hasattr(self, "features_map"):
          self.features_map = {}
        return self


    def _get_mmap(self, mmap_path,  dtype, shape):
      shape[0] = max(shape[0], len(self))
      # what happens when the datastore shrinks??
      ret = np_mmap(mmap_path, dtype, shape)
      if  not hasattr(self, "mmap_access_cnt"): self.mmap_access_cnt=0
      if self.mmap_access_cnt % 100==0: #let's flush intermittently just in case the OS needs to synch.
        ret.flush()
        self.mmap_access_cnt=0
      self.mmap_access_cnt+=1
      return ret

    # we use class variables because we don't want it serialized in an instance of this dataset. this might take up too much memory, so we might use an LRU cache instead.
    igzip_fobj = {}
    def _get_igzip_fobj(self, file_path):
        if file_path in Datastore.igzip_fobj:
            return Datastore.igzip_fobj[file_path]
        Datastore.igzip_fobj[file_path] = fobj = get_file_read_obj(file_path)
        return fobj

    # we use class variables because we don't want it serialized in this instance. this might take up too much memory, so we might use an LRU cache instead.
    db_table = {}
    db_connection = {}
    def _get_db_table(self, table_name, connection_url):
        if (table_name, connection_url) in Datastore.db_table:
            table =  Datastore.db_table[(table_name, connection_url)]
        else:
            if connection_url in Datastore.db_connection:
                db =  Datastore.db_connection[connection_url]
            else:
                Datastore.db_connection[connection_url] = db = DatabaseExt(connection_url)
            Datastore.db_table[(table_name, connection_url)] = table = db[table_name]
        return table

    @staticmethod
    def _add_idx(batch, indices, id_feature,):
        batch[id_feature] = indices # will this be shuffled if we are in shuffled mode?
        return batch

    @staticmethod
    def _move_to_mmap_col(batch, src_feature, id_feature, mmap_path, dtype, shape):
        ret = np_mmap(mmap_path, dtype, shape)
        ret[batch[id_feature]] = batch[dst_feature_view]

    @classmethod
    def from_mmap(cls,  feature_view, shape, mmap_path=None, dtype='float32', dtype_str_len=100, id_feature="id", batch_size=1000, num_proc=4):
      return cls.from_dict({}).add_mmap(feature_view=feature_view, shape=shape, mmap_path=mmap_path, dtype=dtype, dtype_str_len=dtype_str_len, id_feature=id_feature, batch_size=batch_size, num_proc=num_proc)

    def move_to_mmap(self, src_feature, dst_feature_view=None, shape=None, mmap_path=None, dtype='float32', dtype_str_len=100, id_feature="id", batch_size=1000, num_proc=4):
      if dst_feature_view is None:
        self = self.rename_column(src_feature, "__tmp__"+src_feature)
        dst_feature_view = src_feature
        src_feature = "__tmp__"+src_feature
      if shape is None:
        item = self[0][src_feature]
        if type(item) == np.ndarray:
          shape = item.shape
          dtype = item.dtype
        elif type(item) == 'str':
          dtype = 'unicode'
          shape = [-1, max(len(item), dtype_str_len)]
        elif type(item) == 'int':
          dtype = 'int32'
          shape = [-1, 1]
        elif type(item) == 'float':
          dtype = 'float32'
          shape = [-1, 1]
        else:
          raise RuntimeError(f"could not infer shape and dtype for example {item}")
      if shape[0] < len(self):
        shape[0] = len(self)
      self.add_mmap(feature_view=dst_feature_view, shape=shape, mmap_path=mmap_path, dtype=dtype, id_feature=id_feature, batch_size=batch_size, num_proc=num_proc)
      val = self.features_map[dst_feature_view]
      self.map(Datastore._move_to_mmap_col, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'src_feature':src_feature, 'id_feature':id_feature, 'mmap_path': val['mmap_path'], 'dtype': val['dtype'], 'shape':shape})
      return self.remove_columns(src_feature)

    #mapping a feature/columun to a memmap array accessed by row
    def add_mmap(self, feature_view, shape, mmap_path=None, dtype='float32', dtype_str_len=100, id_feature="id", batch_size=1000, num_proc=4):
      if not hasattr(self, 'features_map'): self.features_map = {}
      if hasattr(self, 'id_feature') and self.id_feature != id_feature:
        raise RuntimeError(f"attempting to reset the index to {id_feature}")
      else:
        self.id_feature = id_feature
      if not self.cache_files:
        dataset_path = get_temporary_cache_files_directory()
      else:  
        dataset_path = os.path.dirname(self.cache_files[0]['filename'])
      if mmap_path is None:
          mmap_path = os.path.abspath(os.path.join(dataset_path, feature_view+".mmap"))
      shape = list(shape)
      shape[0] = max(shape[0], len(self))
      if id_feature not in self.features:
        if len(self) == 0 and shape[0] > 0:
            new_self = Datastore.from_dict({id_feature: range(shape[0])})
            new_self.features_map = self.features_map
            self = new_self
            ids = dict([(a,1) for a in range(len(self))])
            self.id_idx_identical = True
        else:
            self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'id_feature': id_feature})
            ids = dict([(a,1) for a in range(len(self))])
            self.id_idx_identical = True
      else:
        ids = dict([(a,1) for a in self[id_feature]])
      missing_ids = []
      for id in range(shape[0]):
          if id not in ids:
            missing_ids.append(id)
      if missing_ids:
            self = self.add_batch({id_feature: missing_ids})
            if not hasattr(self, 'id_idx_identical'):  self.id_idx_identical = True
            if self.id_idx_identical:
              contiguous, start, end = is_contiguous(missing_ids)
              self.id_idx_identical = start ==len(self) and contiguous
            else:
              self.id_idx_identical = False
      if not isinstance(dtype, str):
          dtype =np.dtype(dtype).name
      self.features_map[feature_view] = {'type':"mmap", 'path': mmap_path,  'dtype': dtype, 'shape': shape}
      return self


    @classmethod
    def from_igzip(cls, feature_view, path,  id_feature="id", batch_size=1000, num_proc=4):
      return cls.from_dict({}).add_igzip(feature_view=feature_view, path=path,  id_feature=id_feature, batch_size=batch_size, num_proc=num_proc)

    #mapping a feature/columun to an indexed gzip file accessed by line 
    def add_igzip(self, feature_view, path,  id_feature="id", batch_size=1000, num_proc=4):
      if not hasattr(self, 'features_map'): self.features_map = {}
      if hasattr(self, 'id_feature') and self.id_feature != id_feature:
        raise RuntimeError(f"attempting to reset the index to {id_feature}")
      else:
        self.id_feature = id_feature
      fobj = self._get_igzip_fobj(path)
      if id_feature not in self.features:
          if len(self) == 0:
            new_self = Datastore.from_dict({id_feature: range(len(fobj))})
            new_self.features_map = self.features_map
            self = new_self
            ids = dict([(a,1) for a in range(len(self))])
            self.id_idx_identical = True
          else:
            self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'id_feature': id_feature})
            ids = dict([(a,1) for a in range(len(self))])
            self.id_idx_identical = True
      else:
          ids = dict([(a,1) for a in self[id_feature]])
      missing_ids=[]
      for id in range(len(fobj)):
            if id not in ids:
              missing_ids.append(id)
      if missing_ids:
              self = self.add_batch({id_feature: missing_ids})
              if not hasattr(self, 'id_idx_identical'):  self.id_idx_identical = True
              if self.id_idx_identical:
                contiguous, start, end = is_contiguous(missing_ids)
                self.id_idx_identical = start ==len(self) and contiguous
              else:
                self.id_idx_identical = False
      self.features_map[feature_view] = {'type':"igzip", 'path': path}
      return self

    def move_to_sql(self, src_feature_to_dst_feature_map, table_name=None, connection_url=None,  id_feature="id",  batch_size=1000, num_proc=4):
      if table_name is None:
          #print (self.info.builder_name, self.info.config_name)
          table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}"
      if not connection_url:
          connection_url="sqlite:///"+self.cache_files[0]['filename'].replace(".arrow", ".db")
      table = Datastore._get_db_table(self, table_name, connection_url)

      if dst_feature_view is None:
        self = self.rename_column(src_feature, "__tmp__"+src_feature)
        dst_feature_view = src_feature
        src_feature = "__tmp__"+src_feature
      value = self[0][src_feature]
      if type(value) is str: #we don't want to save as json type just in case
        value="**"

      dtype = table.db.types.guess(value)
      self.add_sql(feature_view=[(dst_feature_view, dtype)], table_name=table_name, connection_url=connection_url, id_feature=id_feature, batch_size=batch_size, num_proc=num_proc)
      if connection_url=="sqlite://": # I don't think we can pass in-memory sqlite table in multi-processing mode, so just do it in single-thread, single-process
        len_self = len(self)
        with Datastore.db_connection[connection_url]:
          for rng in range(0, len_self, batch_size):
            max_rng = min(len_self, rng+batch_size)
            batch = self._getitem(slice(rng, max_rng), format_columns=[id_feature, src_feature])
            Datastore.upsert_sql_from_batch(batch, self.features_map, id_feature, src_feature_to_dst_feature_map,)
      else:
        self.map(Datastore.upsert_sql_from_batch, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'features_map':features_map, 'src_feature_to_dst_feature_map': src_feature_to_dst_feature_map, 'id_feature':id_feature})
      return self.remove_columns(src_feature)

    # mapping one or more columns/features to a sql database. creates a sqlalchmey/dataset dynamically with id_feature as the primary key. 
    # TODO: remember to strip passwords from any connection_url. passwords should be passed as vargs and added to the conneciton url dynamically
    # passwords should not be perisisted.
    # NOTE: this dataset will not automatically change if the database changes. call this method again to sync the size and schema
    def add_sql(self, feature_view=None, table_name=None, connection_url=None, dtype="str", id_feature="id",  batch_size=1000, num_proc=4):
        if table_name is None:
          #print (self.info.builder_name, self.info.config_name)
          table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}"
          #print (table_name)
        if not connection_url:
          connection_url="sqlite:///"+self.cache_files[0]['filename'].replace(".arrow", ".db")
        if type(feature_view) is str:
          feature_view = [(feature_view, dtype)]
        if not hasattr(self, 'features_map'): self.features_map = {}
        if hasattr(self, 'id_feature') and self.id_feature != id_feature:
          raise RuntimeError(f"attempting to reset the index to {id_feature}")
        else:
          self.id_feature = id_feature
        table = self._get_db_table(table_name, connection_url)
        if not feature_view and table.columns:
            feature_view = table.columns
        elif not feature_view:
            raise RuntimeError(f"No feature_view(s) and no column definition for table view {table_name}")
        table_ids = table.find(_columns=id_feature)
        if id_feature not in self.features:
          if len(self) == 0 and table_ids:
            new_self = Datastore.from_dict({id_feature: range(max([id[id_feature] for id in table_ids]))})
            new_self.features_map = self.features_map
            self = new_self
            ids = dict([(a,1) for a in range(len(self))])
            self.id_idx_identical = True
          else:
            self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'id': id_feature})
            ids = dict([(a,1) for a in range(len(self))])
            self.id_idx_identical = True
        else:
          ids = dict([(a,1) for a in self[id_feature]])
        missing_ids = []
        for id in table_ids:
          if id[id_feature] not in ids:
            missing_ids.append(id[id_feature])
        if missing_ids:
            self = self.add_batch({id_feature: missing_ids})
            if not hasattr(self, 'id_idx_identical'):  self.id_idx_identical = True
            if self.id_idx_identical:
              contiguous, start, end = is_contiguous(missing_ids)
              self.id_idx_identical = start ==len(self) and contiguous
            else:
              self.id_idx_identical = False
        for col in feature_view:
            if type(col) is tuple:
              col, dtype = col
            else:
              dtype=None
            if col == id_feature:
                continue
            if col not in table.columns:
              if type(dtype) is str:
                if 'int' in dtype:
                  value = 0
                elif 'float' in dtype:
                  value = 0.0
                else:
                  value = '**'
                dtype = table.db.types.guess(value)
              if dtype is not None:
                table.create_column(col, dtype)
            self.features_map[col] = {'type':'sql', 'connection_url': connection_url, 'table_name': table_name}
        return self
    

    #filter_sql uses a sql query on columns that are mapped to sql. 
    #could be faster than doing a normal "filter"
    #the parameters are the same as the "find" method from dataset.Table.
    #example: dataset.filter_sql(lang='ru') will return those items in the dataset that has the language 'ru'.
    #NOTE: fts not yet working.
    def filter_sql(self, **kwargs):
      if not hasattr(self, 'features_map'): self.features_map = {}
      _fts_q = kwargs.pop('_fts_q', None)
      _limit = kwargs.pop('_limit', None)
      _offset = kwargs.pop('_offset', 0)
      order_by = kwargs.pop('order_by', None)
      _streamed = kwargs.pop('_streamed', False)
      _step = kwargs.pop('_step', QUERY_STEP)
      format_type = kwargs.pop('format_type', None)
      format_kwargs = kwargs.pop('format_kwargs', None)
      if not kwargs:
          raise RuntimeError("no query provided")
      found_table = None
      for key, item in kwargs.item():
        found = False
        for feature_view, val in self.features_map.items():
          if val['type']=='sql':
            table = self._get_db_table(val['table_name'], val['connection_url'])
            if key in table.columns:
              if found_table and table != found_table:
                raise RuntimeError("filtering from multiple sql tables not supported")
              found_table = table
              found = True
              break
        if not found:
          raise RuntimeError(f"found query on a column {key} that is not a sql column")
      kwargs['_fts_q'] = _fts_q
      kwargs['_columns'] = [self.id_feature]
      kwargs['_limit'] = _limit
      kwargs['_offset'] = _offset
      kwargs['order_by'] = order_by
      kwargs['_streamed'] = _streamed
      kwargs['_step'] = _step
      ids = dict([(val['id'],1) for val in found_table.find(*[], **kwargs)])
      if hasattr(self, 'id_idx_identical') and self.id_idx_identical:
        ret = self.select(ids)
        ret.id_idx_identical=False
        return ret
      else:
        return self.filter(lambda example: example['id'] in ids)


    # note that while the id feature corresponds to an item into an external storage, accessing an arrow dataset by datataset[index]
    # will not be guranteed to get the corresponding id. a[0] will return the first item in the current subset of the dataset. 
    # but a[0] does not necessarily return {'id':0, ...}
    # instead, a[0] may return {'id': 10, 'mmap_embed': <array correponding to the 10th location in the mmap file>}. 
    # To get dataset items by 'id', use either filter or filter_sql.
    def _getitem(
        self,
        key: Union[int, slice, str], # should this be list as well??
        format_type=None,
        format_columns=None,
        output_all_columns=False,
        format_kwargs=None,
    ) -> Union[Dict, List]:
        if not hasattr(self, 'features_map'): self.features_map = {}
        # assumine we do error checking re format_columns and output_all_columns at a higher level??
        format_columns = copy.copy(format_columns)
        # this is the case where we are not getting any views.
        if (not self.features_map) or (type(key) is str and key not in self.features_map):
          return super()._getitem(
              key,
              format_type=format_type,
              format_columns=format_columns,
              output_all_columns=output_all_columns,
              format_kwargs=format_kwargs)
        
        # this is the case where there are more than one columns, some of which might
        # be an arrow column and at least one view. For the view, we need to also get the "id".  

        # let's prepare the parameters to get just the arrow portion of the dataset
        orig_key = key
        if type(key) is str:
          if not format_columns:
            format_columns = [key]
          else:
            format_columns.append(key)
          if key in self.features_map:
            key = self.id_feature
        missing=[]
        if format_columns:
            for c in copy.copy(format_columns):
                if c in self.features_map:
                     missing.append(c)
                     format_columns.remove(c)
            if self.id_feature not in format_columns:
                format_columns.append(self.id_feature)
            else:
                missing.append(self.id_feature)

        # let's get the data that is in the arrow portion first, including the id
        outputs = super()._getitem(
              key,
              format_type=format_type,
              format_columns=format_columns,
              output_all_columns=output_all_columns,
              format_kwargs=format_kwargs)

        # this is the case where we are only getting view data, so the only arrow data returned is the 'id'.
        # so we need the id column identified so we can index into the view data source.
        if type(outputs) in (np.array, list):
          outputs = {'id': outputs}

        # do some cleanup.
        if type(orig_key) is str and format_columns and self.id_feature in format_columns:
            format_columns.remove(self.id_feature)
        if format_columns is not None:
            format_columns.extend(missing)
        # now get the views and combine views and  arrow portion 
        return self._format_views(outputs, format_columns=format_columns, format_type=format_type, 
                                 output_all_columns=output_all_columns, format_kwargs=format_kwargs)
        
    def _format_views(self,  
        outputs_or_keys,       
        format_type=None,
        format_columns=None,
        output_all_columns=False,
        format_kwargs=None):

        def getitems(self, outputs, keys, contiguous, start, end, format_columns, output_all_columns, mmap_by_items):
            if not format_columns:
                items = list(self.features_map.items())
            else:
                items = [(column, self.features_map[column]) for column in format_columns if column in self.features_map]
            sql_results = {}
            for feature, val in items:
                if val['type'] == 'mmap':
                    mmap_array = self._get_mmap(val['path'], val['dtype'], val['shape'])
                    if mmap_by_items:
                        if contiguous:
                            outputs[feature] = [mmap_array[i]  for i in range(start, end)]
                        else:
                            outputs[feature] = [mmap_array[i] for i in keys]
                    else:
                        if contiguous:
                            outputs[feature] = mmap_array[start:end]
                        else:
                            outputs[feature] = mmap_array[keys]                            
                elif val['type'] == 'igzip':
                    if contiguous:
                        outputs[feature] = self._get_igzip_fobj(val['path'])[start:end]
                    else:
                        outputs[feature] = self._get_igzip_fobj(val['path'])[keys]
                elif val['type'] == 'sql':
                    sql_results[(val['table_name'], val['connection_url'])] = sql_results.get((val['table_name'], val['connection_url']),[])+[feature]
            for table_connection, features in sql_results.items():
                table_name, connection_url = table_connection
                table= self._get_db_table(table_name, connection_url)
                if contiguous:
                    for row in table.find(**{table._primary_id:{'between': (start, end)}, '_columns':features+['id']}):
                        for feature in features:
                            outputs[feature] = outputs.get(feature,[]) + [row[feature]]
                elif type(keys) is int:
                    row = table.find_one(**{table._primary_id: keys, '_columns':features+['id']})
                    if row:
                        for feature in features:
                            outputs[feature] = row[feature]
                else:
                    for row in table.find(**{table._primary_id:{'in': keys}, '_columns':features+['id']}):
                        for feature in features:
                            outputs[feature] = outputs.get(feature,[]) + [row[feature]]

            return outputs
        format_kwargs = format_kwargs if format_kwargs is not None else {}
        format_columns = format_columns if format_columns is not None else []
        start = end = 0
        contiguous = False
        if format_type in ("custom", "torch", "tensorflow", None) and type(outputs_or_keys) is not pd.DataFrame: 
            transform = format_kwargs.get('transform')
            if isinstance(outputs_or_keys, str):
                keys = outputs_or_keys
                outputs = {}
                contiguous=True
            elif isinstance(outputs_or_keys, slice):
                keys = outputs_or_keys
                outputs = {}
                contiguous=True
            elif isinstance(outputs_or_keys, dict):
                keys = outputs_or_keys[self.id_feature]
                outputs = outputs_or_keys
            else:
                keys = outputs_or_keys
                outputs = {}
            if not contiguous:
                  if isinstance(keys, int):
                        contiguous = False
                  else:
                        contiguous, start, end = is_contiguous(keys)
            else:
                  if isinstance(keys, slice):
                    start = 0 if keys.start is None else keys.start
                    end = len(self) if keys.stop is None else keys.stop
                  else:
                    start = keys[0]
                    end = keys[-1]+1
            outputs = getitems(self, outputs, keys, contiguous, start, end, format_columns, output_all_columns, mmap_by_items=False)
            if transform is not None:
              outputs = transform(outputs)
            if self.id_feature in outputs and format_columns and self.id_feature not in format_columns: del outputs[self.id_feature] 
            # is this right. will custom ever return a dict type if there is only one column, or do we 
            # default to returning the only column.
            if len(outputs) == 1: outputs = list(outputs.values())[0]
            if format_type == "torch":
              import torch
              return torch.tensor(outputs, **format_kwargs)
            elif format_type == "tensorflow":
              import tensorflow
              return tensorflow.ragged.constant(outputs, **format_kwargs)
            else:
              return outputs
        elif format_type == "pandas" or isinstance(outputs_or_keys, pd.DataFrame):
            # do we do transforms for this case??
            if isinstance(outputs_or_keys, str):
                start = 0 
                end = len(self) 
                keys = outputs_or_keys
                outputs = None
                contiguous=True
            elif isinstance(outputs_or_keys, slice):
                start = 0 if outputs_or_keys.start is None else outputs_or_keys.start
                end = len(self) if outputs_or_keys.stop is None else outputs_or_keys.stop
                keys = outputs_or_keys
                outputs = None
                contiguous=True
            elif isinstance(outputs_or_keys, dict) or isinstance(outputs_or_keys,  pd.DataFrame):
                outputs = outputs_or_keys
                outputs = pd.DataFrame(outputs)
                keys = outputs_or_keys[self.id_feature]
                contiguous, start, end = is_contiguous(keys)
            else:
                raise RuntimeError("got unknown outputs or keys type")
            if outputs is None:
                outputs = pd.DataFrame()
            outputs = getitems(self, outputs,  keys, contiguous, start, end, format_columns, output_all_columns, mmap_by_items=True)
            if self.id_feature in outputs and format_columns and self.id_feature not in format_columns: 
              outputs.drop(self.id_feature, axis=1) 
            if len(format_columns) == 1:
              outputs = outputs[format_columns[0]]
            return outputs
        raise RuntimeError("got unknown outputs or keys type")

    @staticmethod
    def upsert_sql_from_batch(batch, features_map, id_feature, src_feature_to_dst_feature_map):
      sql_results={}
      for feature in batch.keys():
        if features_map.get(feature):
          val = features_map[feature]:
          if val['type'] == 'sql':
            sql_results[(val['table_name'], val['connection_url'])] = sql_results.get((val['table_name'], val['connection_url']),[])+[feature]
      for key, features in sql_results.items():
        table_name, connection_url = key
        db = DatabaseExt(connection_url)
        with db:
            table = db[table_name]
            batch2 = [{dst_feature_view: a, id_feature: b} for a, b in zip(batch[src_feature], batch[id_feature])]
            try:
              table.insert_many(batch2)
            except:
              batch2 = [{dst_feature_view: a, id_feature: b} for a, b in zip(batch[src_feature], batch[id_feature])]
              table.update_many(batch2, [id_feature])
            batch2 = None

    @staticmethod
    def map_fn_with_indices_handle_views(batch, indices, map_fn, fn_kwargs, handle_views, features_map, id_feature, remove_columns):
      ret = map_fn(batch, indices, **fn_kwargs)
      if ret is not None:
        if remove_columns:
          for key in remove_columns:
            del ret[key]
        if features_map and id_feature not in ret:
          raise RuntimeError(f"Datstore returned from map must have an {id_feature} column to link to views.")
        for key, val in features_map.items():
          if handle_views == 2:
            if val['type'] == 'mmap':
                mmap_array = np_mmap(val['path'], val['dtype'], val['shape'])
                mmap_array[batch[id_feature]] = batch[feature]                     
            elif val['type'] == 'igzip':
                raise RuntimeError("cannot update an igzip file")
          elif handle_views == 1:
            if key in ret:
              del ret[key]
        Datastore.upsert_sql_from_batch(ret, features_map, id_feature, None)
      return ret

    @staticmethod
    def map_fn_with_handle_views(batch, map_fn, fn_kwargs, handle_views, features_map, id_feature, remove_columns):
      ret = map_fn(batch, **fn_kwargs)
      if ret is not None:
        if remove_columns:
          for key in remove_columns:
            del ret[key]
        if features_map and id_feature not in ret:
          raise RuntimeError(f"Datstore returned from map must have an {id_feature} column to link to views.")
        for key in features_map:
          if handle_views == 2:
            if val['type'] == 'mmap':
                mmap_array = np_mmap(val['path'], val['dtype'], val['shape'])
                mmap_array[batch[id_feature]] = batch[feature]                     
            elif val['type'] == 'igzip':
                raise RuntimeError("cannot update an igzip file")
          elif handle_views == 1:
            if key in ret:
              del ret[key]
        Datastore.upsert_sql_from_batch(ret, features_map, id_feature, None)
      return ret


    PERSIST_IN_ARROW = 0
    STATIC_VIEWS = 1
    UPDATE_VIEWS = 2

    #parameter handle_views tells us how to handle views_data. 
    #PERSIST_IN_ARROW - all data returned will be persisted to arrow storage and not views. this will detach all views.
    #STATIC_VIEWS - keep the views attached to external storage without change. *deafult*
    #UPDATE_VIEWS - update views based on what is returned by the map function. this will create a side-effect.
    #WARNING for caching a result of map with a side-effect will create unexpected results.
    def map(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[List[str]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        handle_views=Datastore.STATIC_VIEWS, 
    ) -> "Datastore":
      if not hasattr(self, 'features_map'): self.features_map = {}
      features_map= copy.deepcopy(self.features_map)
      for column in remove_columns if remove_columns is not None else []:
          if column in features_map:
              del features_map[column]
      if handle_views > 0:
        fn_kwargs = {'fn_kwargs': fn_kwargs, 'features_map': features_map, 'map_fn': function, 'handle_views': handle_views, 'id_feature': self.id_feature, 'remove_columns': remove_columns}
        if with_indices:
          function = Datastore.map_fn_with_indices_handle_views
        else:
          function = Datastore.map_fn_with_handle_views

      ret = super().map(function=function, with_indices=with_indices, input_columns=input_columns,
                     batched=batched, batch_size=batch_size, drop_last_batch=drop_last_batch, 
                     remove_columns=remove_columns, keep_in_memory=keep_in_memory, 
                     load_from_cache_file=load_from_cache_file, cache_file_name=cache_file_name,
                     writer_batch_size=writer_batch_size, features=features,
                     disable_nullable=disable_nullable, fn_kwargs=fn_kwargs,
                     num_proc=num_proc, suffix_template=suffix_template,
                     new_fingerprint=new_fingerprint)

      if ret._fingerprint == self._fingerprint:
        return self
      return Datastore.from_dataset(ret, features_map={} if not keep_views else features_map)

    @transmit_format
    @fingerprint_transform(inplace=False)
    def add_column(self, name: str, column: Union[list, np.array], new_fingerprint: str):
        if not hasattr(self, 'features_map'): self.features_map = {}
        if name in self.features_map:
            raise RuntimeError(f"column {name} is alredy a view")
        ret = super().add_column(name=name, column=column, new_fingerprint=new_fingerprint)
        return Datastore.from_dataset(ret, features_map=self.features_map)

    def class_encode_column(self, column: str) -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        if column in self.features_map:
            raise NotImplementedError()
        ret = super().class_encode_column(column)
        return Datastore.from_dataset(ret, features_map=self.features_map)
    
    @fingerprint_transform(inplace=False)
    def flatten(self, new_fingerprint, max_depth=16) -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        ret = super().flatten(new_fingerprint, max_depth)
        return Datastore.from_dataset(ret, features_map=self.features_map)

    def cast(
        self,
        features: Features,
        batch_size: Optional[int] = 10_000,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 10_000,
        num_proc: Optional[int] = None,
    ) -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        for feature in self.features_map:
            if feature not in features:
                continue
            raise RuntimeError(f"cannot cast a view {feature}")
        ret = super().cast(
          features =features,
          batch_size = batch_size ,
          keep_in_memory = keep_in_memory,
          load_from_cache_file = load_from_cache_file,
          cache_file_name = cache_file_name,
          writer_batch_size = writer_batch_size,
          num_proc = num_proc)
        return Datastore.from_dataset(ret, features_map=self.features_map)

    @fingerprint_transform(inplace=False)
    def remove_columns(self, column_names: Union[str, List[str]], new_fingerprint) -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        features_map= copy.deepcopy(self.features_map)
        if type(column_names) is str: column_names = [column_names] 
        for column in copy.copy(column_names):
            if column in features_map:
                del features_map[column]
                column_names.remove(column)
        if not column_names:
            return Datastore.from_dataset(self, features_map=features_map)
        ret = super().remove_columns(column_names=column_names, new_fingerprint=new_fingerprint)
        return Datastore.from_dataset(ret, features_map=features_map)

    @fingerprint_transform(inplace=False)
    def rename_column(self, original_column_name: str, new_column_name: str, new_fingerprint) -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        features_map= copy.deepcopy(self.features_map)
        if original_column_name in features_map:
            val = features_map[original_column_name]
            del features_map[original_column_name]
            features_map[new_column_name] = val
            return Datastore.from_dataset(self, features_map=features_map)
        ret = super().rename_column(original_column_name=original_column_name, new_column_name=new_column_name, new_fingerprint=new_fingerprint)
        return Datastore.from_dataset(self, features_map=features_map)
        
    @fingerprint_transform(inplace=False)
    def rename_columns(self, column_mapping: Dict[str, str], new_fingerprint)  -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        features_map= copy.deepcopy(self.features_map)
        for original_column_name, new_column_name in list(column_mapping.items()):
            val = features_map[original_column_name]
            del features_map[original_column_name]
            features_map[new_column_name] = val
            del column_mapping[original_column_name]
        if not column_mapping:
          return Datastore.from_dataset(self, features_map=features_map) 
        ret = super().rename_column(column_mapping=column_mapping, new_fingerprint=new_fingerprint)
        return Datastore.from_dataset(ret, features_map=features_map)

    def prepare_for_task(self, task: Union[str, TaskTemplate]) -> "Dataset":
        if not hasattr(self, 'features_map'): self.features_map = {}
        ret = super().prepare_for_task(task)
        return Datastore.from_dataset(ret, features_map=features_map)
    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["load_from_cache_file", "cache_file_name"])
    def filter(
        self,
        function: Optional[Callable] = None,
        with_indices=False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batch_size: Optional[int] = 1000,
        remove_columns: Optional[List[str]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":
        ret = super().filter(
            function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batch_size=batch_size,
            remove_columns=remove_columns,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            fn_kwargs=fn_kwargs,
            num_proc=num_proc,
            suffix_template=suffix_template,
            new_fingerprint=new_fingerprint)
        if not hasattr(self, 'features_map'): self.features_map = {}
        features_map= copy.deepcopy(self.features_map)
        for column in remove_columns if remove_columns is not None else []:
            if column in features_map:
                del features_map[column]
        return Datastore.from_dataset(ret, features_map=features_map)

    #replayable_table_alteration
    @fingerprint_transform(inplace=False, ignore_kwargs=["cache_file_name"])
    def flatten_indices(
        self,
        keep_in_memory: bool = False,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = True,
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":
        ret = super().flatten_indices(
            keep_in_memory=keep_in_memory,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            new_fingerprint=new_fingerprint,
            )
        if not hasattr(self, 'features_map'): self.features_map = {}
        return Datastore.from_dataset(ret, features_map=self.features_map)

    
    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["indices_cache_file_name"])
    def select(
        self,
        indices: Iterable,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        ret = super().select(
            indices=indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
            ) 

        return Datastore.from_dataset(ret, features_map=self.features_map)


    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["load_from_cache_file", "indices_cache_file_name"])
    def sort(
        self,
        column: str,
        reverse: bool = False,
        kind: str = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        if column in self.features_map:
            raise NotImplementedError()
        ret = super().sort(
            column=column,
            reverse=reverse,
            kind=kind,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
        )
        return Datastore.from_dataset(ret, features_map=self.features_map)



    @transmit_format
    @fingerprint_transform(
        inplace=False, randomized_function=True, ignore_kwargs=["load_from_cache_file", "indices_cache_file_name"]
    )
    def shuffle(
        self,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        ret = super().shuffle(
            seed=seed,
            generator=generator,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
            )
        return Datastore.from_dataset(ret, features_map=self.features_map)
  
    @transmit_format
    @fingerprint_transform(
        inplace=False,
        randomized_function=True,
        fingerprint_names=["train_new_fingerprint", "test_new_fingerprint"],
        ignore_kwargs=["load_from_cache_file", "train_indices_cache_file_name", "test_indices_cache_file_name"],
    )
    def train_test_split(
        self,
        test_size: Union[float, int, None] = None,
        train_size: Union[float, int, None] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = True,
        train_indices_cache_file_name: Optional[str] = None,
        test_indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        train_new_fingerprint: Optional[str] = None,
        test_new_fingerprint: Optional[str] = None,
    ) -> "DatasetDict":
        if not hasattr(self, 'features_map'): self.features_map = {}
        ret = super.train_test_split(
            test_size=test_size,
            train_size=train_size,
            shuffle=shuffle,
            seed=seed,
            generator=generator,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            train_indices_cache_file_name=train_indices_cache_file_name,
            test_indices_cache_file_name=test_indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            train_new_fingerprint=train_new_fingerprint,
            test_new_fingerprint=test_new_fingerprint,
        )
        for key in list(ret.keys()):
            ret[key] = Datastore.from_dataset(ret, features_map=self.features_map)
        return ret


    @transmit_format
    @fingerprint_transform(inplace=False)
    def add_item(self, item: dict, new_fingerprint: str):
        if not hasattr(self, 'features_map'): self.features_map = {}
        ret = super().add_item(item=item,
          new_fingerprint=new_fingerprint)
        return Datastore.from_dataset(ret, features_map=self.features_map)

    @transmit_format
    @fingerprint_transform(inplace=False)
    def add_batch(self, batch: dict, new_fingerprint: str):
        """Add batch to Dataset.

        Args:
            batch (dict): batch data to be added.

        Returns:
            :class:`Dataset`
        """
        if not hasattr(self, 'features_map'): self.features_map = {}
        keys = list(batch.keys())
        len_batch = len(batch[keys[0]])
        features = list(self.features)
        for feature in self.features:
          if feature not in keys:
            batch[feature] = [None]*len_batch
        item_table = InMemoryTable.from_pydict(batch)
        # Cast batch
        schema = pa.schema(self.features.type)
        item_table = item_table.cast(schema)
        # Concatenate tables
        table = concat_tables([self._data, item_table])
        if self._indices is None:
            indices_table = None
        else:
            item_indices_array = pa.array(list(range(len(self._data), len(self._data)+len(item_table._data))), type=pa.uint64())
            item_indices_table = InMemoryTable.from_arrays([item_indices_array], names=["indices"])
            indices_table = concat_tables([self._indices, item_indices_table])
        ret=Dataset(
            table,
            info=self.info.copy(),
            split=self.split,
            indices_table=indices_table,
            fingerprint=new_fingerprint,
        )
        return Datastore.from_dataset(ret, features_map=self.features_map)

    def align_labels_with_mapping(self, label2id: Dict, label_column: str) -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        ret = super().align_labels_with_mapping(label2id=label2id,
            label_column=label_column)
        return Datastore.from_dataset(ret, features_map=self.features_map)
        
    @staticmethod
    def from_csv(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        **kwargs,
    ):
        """Create Datastore from CSV file(s).
        Args:
            path_or_paths (path-like or list of path-like): Path(s) of the CSV file(s).
            split (:class:`NamedSplit`, optional): Split name to be assigned to the dataset.
            features (:class:`Features`, optional): Dataset features.
            cache_dir (:obj:`str`, optional, default ``"~/.cache/huggingface/datasets"``): Directory to cache data.
            keep_in_memory (:obj:`bool`, default ``False``): Whether to copy the data in-memory.
            **kwargs: Keyword arguments to be passed to :meth:`pandas.read_csv`.
        Returns:
            :class:`Datastore`
        """
        # Dynamic import to avoid circular dependency
        from .io.csv import CsvDatasetReader

        return Datastore.from_dataset(CsvDatasetReader(
            path_or_paths, split=split, features=features, cache_dir=cache_dir, keep_in_memory=keep_in_memory, **kwargs
        ).read())

    @staticmethod
    def from_json(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        field: Optional[str] = None,
        **kwargs,
    ):
        """Create Datastore from JSON or JSON Lines file(s).
        Args:
            path_or_paths (path-like or list of path-like): Path(s) of the JSON or JSON Lines file(s).
            split (:class:`NamedSplit`, optional): Split name to be assigned to the dataset.
            features (:class:`Features`, optional): Dataset features.
            cache_dir (:obj:`str`, optional, default ``"~/.cache/huggingface/datasets"``): Directory to cache data.
            keep_in_memory (:obj:`bool`, default ``False``): Whether to copy the data in-memory.
            field (:obj:`str`, optional): Field name of the JSON file where the dataset is contained in.
            **kwargs: Keyword arguments to be passed to :class:`JsonConfig`.
        Returns:
            :class:`Datastore`
        """
        # Dynamic import to avoid circular dependency
        from .io.json import JsonDatasetReader

        return Datastore.from_dataset(JsonDatasetReader(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            field=field,
            **kwargs,
        ).read())

    @staticmethod
    def from_parquet(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        columns: Optional[List[str]] = None,
        **kwargs,
    ):
        """Create Datastore from Parquet file(s).
        Args:
            path_or_paths (path-like or list of path-like): Path(s) of the Parquet file(s).
            split (:class:`NamedSplit`, optional): Split name to be assigned to the dataset.
            features (:class:`Features`, optional): Dataset features.
            cache_dir (:obj:`str`, optional, default ``"~/.cache/huggingface/datasets"``): Directory to cache data.
            keep_in_memory (:obj:`bool`, default ``False``): Whether to copy the data in-memory.
            columns (:obj:`List[str]`, optional): If not None, only these columns will be read from the file.
                A column name may be a prefix of a nested field, e.g. 'a' will select
                'a.b', 'a.c', and 'a.d.e'.
            **kwargs: Keyword arguments to be passed to :class:`ParquetConfig`.
        Returns:
            :class:`Datastore`
        """
        # Dynamic import to avoid circular dependency
        from .io.parquet import ParquetDatasetReader

        return Datastore.from_dataset(ParquetDatasetReader(
            path_or_paths,
            split=split,
            features=features,
            cache_dir=cache_dir,
            keep_in_memory=keep_in_memory,
            columns=columns,
            **kwargs,
        ).read())

    @staticmethod
    def from_text(
        path_or_paths: Union[PathLike, List[PathLike]],
        split: Optional[NamedSplit] = None,
        features: Optional[Features] = None,
        cache_dir: str = None,
        keep_in_memory: bool = False,
        **kwargs,
    ):
        """Create Datastore from text file(s).
        Args:
            path_or_paths (path-like or list of path-like): Path(s) of the text file(s).
            split (:class:`NamedSplit`, optional): Split name to be assigned to the dataset.
            features (:class:`Features`, optional): Dataset features.
            cache_dir (:obj:`str`, optional, default ``"~/.cache/huggingface/datasets"``): Directory to cache data.
            keep_in_memory (:obj:`bool`, default ``False``): Whether to copy the data in-memory.
            **kwargs: Keyword arguments to be passed to :class:`TextConfig`.
        Returns:
            :class:`Datastore`
        """
        # Dynamic import to avoid circular dependency
        from .io.text import TextDatasetReader

        return Datastore.from_dataset(TextDatasetReader(
            path_or_paths, split=split, features=features, cache_dir=cache_dir, keep_in_memory=keep_in_memory, **kwargs
        ).read())


    def save_to_disk(self, dataset_path: str, fs=None, move_files=True):
      # move_files means delete the old files as we create the new files.
        """
        Saves a dataset to a dataset directory, or in a filesystem using either :class:`~filesystems.S3FileSystem` or
        any implementation of ``fsspec.spec.AbstractFileSystem``.
        Note regarding sliced datasets:
        If you sliced the dataset in some way (using shard, train_test_split or select for example), then an indices mapping
        is added to avoid having to rewrite a new arrow Table (save time + disk/memory usage).
        It maps the indices used by __getitem__ to the right rows if the arrow Table.
        By default save_to_disk does save the full dataset table + the mapping.
        If you want to only save the shard of the dataset instead of the original arrow file and the indices,
        then you have to call :func:`datasets.Dataset.flatten_indices` before saving.
        This will create a new arrow table by using the right rows of the original table.
        Args:
            dataset_path (:obj:`str`): Path (e.g. `dataset/train`) or remote URI (e.g. `s3://my-bucket/dataset/train`)
                of the dataset directory where the dataset will be saved to.
            fs (:class:`~filesystems.S3FileSystem`, ``fsspec.spec.AbstractFileSystem``, optional, defaults ``None``):
                Instance of the remote filesystem used to download the files from.
        """
        assert (
            not self.list_indexes()
        ), "please remove all the indexes using `dataset.drop_index` before saving a dataset"

        if is_remote_filesystem(fs):
            dataset_path = extract_path_from_uri(dataset_path)
        else:
            fs = fsspec.filesystem("file")
            cache_files_paths = [Path(cache_filename["filename"]) for cache_filename in self.cache_files]
            # Check that the dataset doesn't overwrite iself. It can cause a permission error on Windows and a segfault on linux.
            if Path(dataset_path, config.DATASET_ARROW_FILENAME) in cache_files_paths:
                raise PermissionError(
                    f"Tried to overwrite {Path(dataset_path, config.DATASET_ARROW_FILENAME)} but a dataset can't overwrite itself."
                )
            if Path(dataset_path, config.DATASET_INDICES_FILENAME) in cache_files_paths:
                raise PermissionError(
                    f"Tried to overwrite {Path(dataset_path, config.DATASET_INDICES_FILENAME)} but a dataset can't overwrite itself."
                )

        # Get json serializable state
        state = {
            key: self.__dict__[key]
            for key in [
                "_fingerprint",
                "_format_columns",
                "_format_kwargs",
                "_format_type",
                "_indexes",
                "_output_all_columns",
                "features_map",
                "id_idx_identical"
                "id_feature"
            ]
        }

        split = self.__dict__["_split"]
        state["_split"] = str(split) if split is not None else split

        state["_data_files"] = [{"filename": config.DATASET_ARROW_FILENAME}]
        state["_indices_data_files"] = (
            [{"filename": config.DATASET_INDICES_FILENAME}] if self._indices is not None else None
        )
        for k in state["_format_kwargs"].keys():
            try:
                json.dumps(state["_format_kwargs"][k])
            except TypeError as e:
                raise TypeError(str(e) + f"\nThe format kwargs must be JSON serializable, but key '{k}' isn't.")

        # Get json serializable dataset info
        dataset_info = asdict(self._info)

        # Save dataset + indices + state + info
        fs.makedirs(dataset_path, exist_ok=True)
        with fs.open(Path(dataset_path, config.DATASET_ARROW_FILENAME).as_posix(), "wb") as dataset_file:
            with ArrowWriter(stream=dataset_file) as writer:
                writer.write_table(self._data)
                writer.finalize()
        if move_files:
          for data_file in self._data_files:
            os.unlink(data_file)
          for cache_filename in self.cache_files:
            os.unlink(cache_filename)
        if self._indices is not None:
            with fs.open(Path(dataset_path, config.DATASET_INDICES_FILENAME).as_posix(), "wb") as indices_file:
                with ArrowWriter(stream=indices_file) as writer:
                    writer.write_table(self._indices)
                    writer.finalize()
        if move_files:
          for indices_data_file in self._indices_data_files:
            os.unlink(indices_data_file)
        with fs.open(
            Path(dataset_path, config.DATASET_STATE_JSON_FILENAME).as_posix(), "w", encoding="utf-8"
        ) as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)
        with fs.open(
            Path(dataset_path, config.DATASET_INFO_FILENAME).as_posix(), "w", encoding="utf-8"
        ) as dataset_info_file:
            # Sort only the first level of keys, or we might shuffle fields of nested features if we use sort_keys=True
            sorted_keys_dataset_info = {key: dataset_info[key] for key in sorted(dataset_info)}
            json.dump(sorted_keys_dataset_info, dataset_info_file, indent=2)
        logger.info("Dataset saved in {}".format(dataset_path))
        for key, value in list(self.features_map.items()):
            # Copy or move file to destination directory
            if 'connection_url' in value:
              if "sqlite:///" in value['connection_url']:
                src = value['connection_url'].replace("sqlite:///", "")
              else:
                continue
            else:
              src = value['path']
            filename = Path(src).name
            dest = os.path.join(dataset_path, filename)
            # if the src is not under the 
            if src != dest and os.path.exists(src):
                if move_files:
                  shutil.move(src, dest)
                else:
                  shutil.copy(src, dest)
                if value['type'] == 'igzip':
                  src = src.replace(".gz", ".igz")
                  dest = dest.replace(".gz", ".igz")
                  if move_files:
                    shutil.move(src, dest)
                  else:
                    shutil.copy(src, dest)
        if move_files:
          return Datastore.load_from_disk(dataset_path, fs=fs,)
        else:
          return self


    #todo, do periodic sync with the shared drive, and lazy loading of shards from shared drive
    @staticmethod
    def load_from_disk(dataset_path: str, fs=None, keep_in_memory: Optional[bool] = None) -> "Datastore":
      # TODO, move from shared drive to cached drive
        ret = Dataset.load_from_disk(dataset_path=dataset_path, fs=fs, keep_in_memory=keep_in_memory)
        with open(
            Path(dataset_path, config.DATASET_STATE_JSON_FILENAME).as_posix(), "r", encoding="utf-8"
        ) as state_file:
            state = json.load(state_file)
        ret.features_map =  state.get("features_map")
        ret.id_idx_identical =  state.get("id_idx_identical")
        fs = fsspec.filesystem("file") if fs is None else fs
        for key, values in list(ret.features_map.items()):
            if 'connection_url' in value:
              if "sqlite:///" in value['connection_url']:
                src = value['connection_url'].replace("sqlite:///", "")
              else:
                continue
            else:
              src = value['path']
            if is_remote_filesystem(fs):
                data_path = os.path.join(dataset_path, src)
                src_dataset_path = fs.path.join(extract_path_from_uri(data_path), 
                tmp_dir = tempfile.TemporaryDirectory()
                data_path = Path(tmp_dir.name, src_dataset_path)
                fs.download(src_dataset_path, data_path.as_posix(), recursive=True)
                if value['type'] == 'igzip':
                  src_dataset_path2 = src_dataset_path2.replace(".gz", ".igz")
                  data_path2 = Path(tmp_dir.name, src_dataset_path2)
                  fs.download(src_dataset_path2, data_path2.as_posix(), recursive=True)
            else:
                data_path = os.path.abspath(os.path.join(dataset_path, src))
            if 'connection_url' in value:
              ret.features_map[key]['connection_url'] =  "sqlite:///"+data_path
            else:
              ret.features_map[key]['path'] =  data_path
        return Datastore.from_dataset(ret)

if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    from datasets import load_dataset
    import fsspec, requests, aiohttp
    def get_oscar_urls(language, shuffled="unshuffled", deduplicated="deduplicated"):
      _BASE_DATA_URL_FORMAT_STR = ("https://s3.amazonaws.com/datasets.huggingface.co/oscar/1.0/{shuffled}/{deduplicated}/{language}/")
      _BASE_CHECKSUM_FILE_NAME = "{language}_sha256.txt"
      base_data_url = _BASE_DATA_URL_FORMAT_STR.format(
                shuffled=shuffled, language=language, deduplicated=deduplicated
            )
      checksum_url = base_data_url + _BASE_CHECKSUM_FILE_NAME.format(language=language)
      with fsspec.open(checksum_url, encoding="utf-8") as f:
        data_filenames = [line.decode().split("\t")[0] for line in f if line]
        return [base_data_url + data_filename for data_filename in data_filenames]

    def download_urls(urls):
      for url in urls:
        if not os.path.exists(url.split("/")[-1]):
          os.system(f"wget {url}")
        data = Datastore.from_dict({}).add_igzip("text", url.split("/")[-1])
        print (data[-1])
      
    args = sys.argv[1:]
    if "-test_igzip" in args:
      if not os.path.exists("wikitext-2"):
        !wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
        !unzip wikitext-2-v1.zip
        !gzip wikitext-2/wiki.train.tokens
      datastore = Datastore.from_igzip("txt", "wikitext-2/wiki.train.tokens.gz")
      datastore = Datastore.from_dict({'id':range(10000), 'len':[5]*10000})
      datastore = datastore.add_igzip("txt", "wikitext-2/wiki.train.tokens.gz")
      print (datastore)
      print (datastore[0])
      print (datastore[10100])
    if "-test_memmap" in args:
       datastore = Datastore.from_mmap('embed', [1000, 512, 512], )
       print (datastore)
       datastore = Datastore.from_dataset(load_dataset("oscar", "unshuffled_deduplicated_sw")['train'])
       datastore = datastore.add_mmap('embed', [-1, 512, 512], )
       datastore = datastore.add_mmap('token', [-1, 512], dtype=np.int32)
       assert (datastore['embed'][0].shape) == (512, 512)
       datastore['embed'][0][0] = 0.0
       assert np.mean(datastore['embed'][0][0]) == 0
       datastore['embed'][0][0] = 1.0
       assert np.mean(datastore['embed'][0][0]) == 1.0
       assert set(datastore[0].keys()) == set(['id', 'text', 'embed', 'token'])
       assert len(datastore['text']) == 24803
       assert len(datastore[0:10]['text']) == 10
       assert (datastore[0:10]['token'].shape) == (10, 512)
       print (datastore)
    if "-test_sql" in args:
       datastore = Datastore.from_dataset(load_dataset("oscar", "unshuffled_deduplicated_sw")['train'])
       datastore= datastore.add_sql('text2')
       datastore = Datastore.from_dataset(load_dataset("oscar", "unshuffled_deduplicated_yo")['train'])
       datastore= datastore.move_to_sql('text','text2')
      

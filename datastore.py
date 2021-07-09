#Copyright July 2021 Ontocord LLC. Licensed under Apache v2 https://www.apache.org/licenses/LICENSE-2.0
#datastore.py

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
### transported as a file transfer with an arrow dataset. So if we
### have <signature>.arrow, we may have fts_<signature>.db (for full
### text indexing) and db_<signature>.db (sqlite database), and
### <siganture>.mmap (mmap file reprsenting a tensor), and
### <singature>.igz (if we wish to store some portion of the text
### columns in igzip format for compression and legacy purposes.


### A note about naming: datasets uses the terms for features and columns interchangably.

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


class Datastore(Dataset): #, dict
    """
    A class that wraps a Huggingface arrow based Dataset to provide some optimized reading and *writing* in various persistance backends. 
    Currently provides support for features bound to memmap, igzip file, and sqlalchemy databases.
    """
    @property 
    def features(self):
        ret = FeaturesWithViews(self._info.features)
        ret.features_map = {} if not hasattr(self, "features_map") else self.features_map
        return ret
        
    def __repr__(self):
        return f"Datastore({{\n    features: {self.features},\n    num_rows: {self.num_rows}\n}})"
        
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
        if  hasattr(dataset, "shared_dir"):
          self.shared_dir=shared_dir
        if shared_dir is not None:
          self.shared_dir = shared_dir
        if not hasattr(self, "shared_dir"):
          self.shared_dir = {}
        return self

                             
    def _get_mmap(self, mmap_path,  dtype, shape):
      if shape[0] < len(self):
          shape[0] = len(self)
      # what happens when the datastore shrinks??
      ret = np_mmap(mmap_path, dtype, shape)
      if self.mmap_access_cnt % 100==0: #let's flush intermittently just in case the OS needs to synch.
        ret.flush()
        self.mmap_access_cnt=0
      self.mmap_access_cnt+=1
      return ret

    # we use class variables because we don't want it serialized in an instance of this dataset
    igzip_fobj = {}
    def _get_igzip_fobj(self, file_path):
        if file_path in Datastore.igzip_fobj:
            return Datastore.igzip_fobj[file_path]
        Datastore.igzip_fobj[file_path] = fobj = get_file_read_obj(file_path)
        return fobj

    # we use class variables because we don't want it serialized in this instance
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

    # todo, change the id key from "id" to something custom. this needs to be stored in the table meta-data.
    @staticmethod
    def _add_idx(batch, indices, idx_feature,):
        batch[idx_feature] = indices # will this be shuffled if we are in shuffled mode?
        return batch

    @staticmethod
    def _move_to_mmap_col(batch, src_feature, idx_feature, mmap_path, dtype, shape):
        ret = np_mmap(mmap_path, dtype, shape)
        ret[batch[idx_feature]] = batch[dst_feature_view]

        
    def move_to_mmap(self, src_feature, dst_feature_view, shape, mmap_path=None, dtype='float32', dtype_str_len=1000, idx_feature="id", batch_size=100000, num_proc=4):
      self.add_mmap(feature_view=dst_feature_view, shape=shape, mmap_path=mmap_path, dtype=dtype, idx_feature=idx_feature, batch_size=batch_size, num_proc=num_proc)
      if shape[0] < len(self):
        shape[0] = len(self)
      val = self.features_map[dst_feature_view]
      self.map(Datastore._move_to_mmap_col, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'src_feature':src_feature, 'idx_feature':idx_feature, 'mmap_path': val['mmap_path'], 'dtype': val['dtype'], 'shape':shape})
      return self.remove_columns(src_feature)

    #mapping a feature/columun to a memmap array accessed by row
    def add_mmap(self, feature_view, shape, mmap_path=None, dtype='float32', dtype_str_len=1000, idx_feature="id", batch_size=100000, num_proc=4):
      if not hasattr(self, 'features_map'): self.features_map = {}
      dataset_path = os.path.dirname(self.cache_files[0]['filename'])
      if mmap_path is None:
               mmap_path = os.path.abspath(os.path.join(dataset_path, feature_view+".mmap"))
      shape = list(shape)
      shape[0] = max(shape[0], len(self))
      if idx_feature not in self.features:
        self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'idx_feature': idx_feature})
        ids = dict([(a,1) for a in range(len(self))])
      else:
        ids = dict([(a,1) for a in self[idx_feature]])
      missing_ids = []
      for id in range(shape[0]):
          if id not in ids:
            missing_ids.append(id)
      if missing_ids:
            self = self.add_batch({idx_feature: missing_ids})
      if not isinstance(dtype, str):
          dtype =np.dtype(dtype).name
      self.features_map[feature_view] = {'type':"mmap", 'path': mmap_path,  'dtype': dtype, 'shape': shape}
      return self

    #mapping a feature/columun to an indexed gzip file accesed by line 
    def add_igzip(self, feature_view, path,  idx_feature="id", batch_size=100000, num_proc=4):
      if not hasattr(self, 'features_map'): self.features_map = {}
      fobj = self._get_igzip_fobj(path)
      if idx_feature not in self.features:
        self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'idx_feature': idx_feature})
        ids = dict([(a,1) for a in range(len(self))])
      else:
        ids = dict([(a,1) for a in self[idx_feature]])
      missing_ids = []
      for id in range(len(fobj)):
          if id not in ids:
            missing_ids.append(id)
      if missing_ids:
            self = self.add_batch({idx_feature: missing_ids})
      self.features_map[feature_view] = {'type':"igzip", 'path': path}
      return self

    @staticmethod
    def _move_to_sql_col(batch, table_name, connection_url, src_feature, dst_feature_view, idx_feature):
          db = DatabaseExt(connection_url)
          with db:
            table = db[table_name]
            batch = [{dst_feature_view: a, idx_feature: b} for a, b in zip(batch[src_feature], batch[idx_feature])]
            try:
              table.insert_many(batch)
            except:
              table.upsert_many(batch, [idx_feature])

    def move_to_sql(self, src_feature, dst_feature_view, table_name=None, connection_url=None,  idx_feature="id",  batch_size=100000, num_proc=4):
      if table_name is None:
          #print (self.info.builder_name, self.info.config_name)
          table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}"
      self.add_sql(feature_view=dst_feature_view, table_name=table_name, connection_url=connection_url, idx_feature=idx_feature, batch_size=batch_size, num_proc=num_proc)
      if not connection_url:
          connection_url="sqlite:///"+self.cache_files[0]['filename'].replace(".arrow", ".db")
      if connection_url=="sqlite://":
        table = Datastore._get_db_table(self, table_name, connection_url)
        len_self = len(self)
        for rng in range(0, len_self, batch_size):
          max_rng = min(len_self, rng+batch_size)
          batch = self._getitem(slice(rng, max_rng), format_columns=[idx_feature, src_feature])
          batch = [{dst_feature_view: a, idx_feature: b} for a, b in zip(batch[src_feature], batch[idx_feature])]
          try:
            table.insert_many(batch)
          except:
            table.upsert_many(batch, [idx_feature])
          batch2 = batch = None
      else:
        self.map(Datastore._move_to_sql_col, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'table_name':table_name, 'connection_url':connection_url, 'src_feature': src_feature, 'dst_feature_view': dst_feature_view, 'idx_feature':idx_feature})
      return self.remove_columns(src_feature)

    # mapping one or more columns/features to a sql database. creates a sqlalchmey/dataset dynamically with idx_feature as the primary key. 
    def add_sql(self, feature_view=None, table_name=None, connection_url=None,  idx_feature="id",  batch_size=100000, num_proc=4):
        if table_name is None:
          #print (self.info.builder_name, self.info.config_name)
          table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}"
          #print (table_name)
        if not connection_url:
          connection_url="sqlite:///"+self.cache_files[0]['filename'].replace(".arrow", ".db")
        if type(feature_view) is str:
          feature_view = [feature_view]
        if not hasattr(self, 'features_map'): self.features_map = {}
        table = self._get_db_table(table_name, connection_url)
        if not feature_view and table.columns:
            feature_view = table.columns
        elif not feature_view:
            raise RuntimeError(f"No feature_view(s) and no column definition for table view {table_name}")
        for col in feature_view:
            if col == idx_feature:
                continue
            if col in self.features:
                raise RuntimeError(f"Feature {col} already in the dataset")
        if idx_feature not in self.features:
          self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'id': idx_feature})
          ids = dict([(a,1) for a in range(len(self))])
        else:
          ids = dict([(a,1) for a in self[idx_feature]])
        missing_ids = []
        for id in table.find(_columns=idx_feature):
          if id[idx_feature] not in ids:
            missing_ids.append(id[idx_feature])
        if missing_ids:
            self = self.add_batch({idx_feature: missing_ids})
        for col in feature_view:
            if col == idx_feature:
                continue
            self.features_map[col] = {'type':'sql', 'connection_url': connection_url, 'table_name': table_name}
        return self
    
    # note that while the id feature corresponds to an index into an external storage, accessing an arrow dataset by index
    # will not be guranteed to get the corresponding id. a[0] will return the first item in the current subset of the dataset. 
    # but a[0] may return {'id': 10, 'mmap_embed': <array correponding to the 10th location in the mmap file>}
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
            key = "id"
        missing=[]
        if format_columns:
            for c in copy.copy(format_columns):
                if c in self.features_map:
                     missing.append(c)
                     format_columns.remove(c)
            if "id" not in format_columns:
                format_columns.append("id")
            else:
                missing.append("id")

        # let's get the data that is in the arrow data first, including the id
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
        if type(orig_key) is str and format_columns and "id" in format_columns:
            format_columns.remove("id")
        if format_columns is not None:
            format_columns.extend(missing)
        # now get the views and combine views and  arrow data 
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
                    if mmap_by_items:
                        if contiguous:
                            outputs[feature] = [ self._get_mmap(val['path'], val['dtype'], val['shape']) for i in range(start, end)]
                        else:
                            outputs[feature] = [ self._get_mmap(val['path'], val['dtype'], val['shape']) for i in keys]
                    else:
                        if contiguous:
                            outputs[feature] = self._get_mmap(val['path'], val['dtype'], val['shape'])[start:end]
                        else:
                            outputs[feature] = self._get_mmap(val['path'], val['dtype'], val['shape'])[keys]                            
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
                keys = slice(0, len(self))
                outputs = {}
                contiguous=True
            elif isinstance(outputs_or_keys, slice):
                keys = outputs_or_keys
                outputs = {}
                contiguous=True
            elif isinstance(outputs_or_keys, dict):
                keys = outputs_or_keys["id"]
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
            if "id" in outputs and format_columns and "id" not in format_columns: del outputs["id"] 
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
                keys = range(start, stop)
                outputs = None
                contiguous=True
            elif isinstance(outputs_or_keys, slice):
                start = 0 if outputs_or_keys.start is None else outputs_or_keys.start
                end = len(self) if outputs_or_keys.stop is None else outputs_or_keys.stop
                keys = range(outputs_or_keys.start, outputs_or_keys.stop)
                outputs = None
                contiguous=True
            elif isinstance(outputs_or_keys, dict) or isinstance(outputs_or_keys,  pd.DataFrame):
                outputs = outputs_or_keys
                outputs = pd.DataFrame(outputs)
                keys = outputs_or_keys["id"]
                contiguous, start, end = is_contiguous(keys)
            else:
                raise RuntimeError("got unknown outputs or keys type")
            if outputs is None:
                outputs = pd.DataFrame()
            outputs = getitems(self, outputs,  keys, contiguous, start, end, format_columns, output_all_columns, mmap_by_items=True)
            if "id" in outputs and format_columns and "id" not in format_columns: 
              outputs.drop("id", axis=1) 
            if len(format_columns) == 1:
              outputs = outputs[format_columns[0]]
            return outputs
        raise RuntimeError("got unknown outputs or keys type")

    def to_csv(
        self,
        path_or_buf: Union[PathLike, BinaryIO],
        batch_size: Optional[int] = None,
        **to_csv_kwargs,
    ) -> int:
      pass

    def to_dict(self, batch_size: Optional[int] = None, batched: bool = False) -> Union[dict, Iterator[dict]]:
        if (not hasattr(self, "features_map") or not self.features_map) and len(self.features) == 1 and "id" in self.features:
            return {}
        #TODO - put back direct mmap access method here?
        ret = super().to_dict(batch_size=batch_size, batched=batched)
        if isinstance(ret, Iterator):
            for r in ret:
                yield self._format_views(r, contiguous=True)
            return 
        return self._format_views(ret, contiguous=True)

    def to_pandas(
        self, batch_size: Optional[int] = None, batched: bool = False
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        if (not hasattr(self, "features_map") or not self.features_map) and len(self.features) == 1 and "id" in self.features:
            return pd.DataFrame()
        #TODO - put back direct mmap access method here?
        ret = super().to_pandas(batch_size=batch_size, batched=batched)
        if isinstance(ret, Iterator):
            for r in ret:
                yield self._format_views(r, contiguous=True)
        return self._format_views(ret, contiguous=True)
        

        
    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["load_from_cache_file", "cache_file_name"])
    def _map_single_old(
        self,
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[List[str]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        new_fingerprint: Optional[str] = None,
        rank: Optional[int] = None,
        offset: int = 0,
        desc: Optional[str] = None,
    ) -> "Datastore":
      if not hasattr(self, 'features_map'): self.features_map = {}
      ret = super()._map_single(function=function,
            with_indices=with_indices,
            input_columns=input_columns,
            batched=batched,
            batch_size=batch_size,
            drop_last_batch=drop_last_batch,
            remove_columns=remove_columns,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            fn_kwargs=fn_kwargs,
            new_fingerprint=new_fingerprint,
            rank=rank,
            offset=offset,
            desc=desc,)
      features_map= copy.deepcopy(self.features_map)
      for column in remove_columns if remove_columns is not None else []:
          if column in features_map:
              del features_map[column]
      return Datastore.from_dataset(ret, features_map=features_map)

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
    ) -> "Datastore":
      if not hasattr(self, 'features_map'): self.features_map = {}
      ret = super().map(function=function, with_indices=with_indices, input_columns=input_columns,
                     batched=batched, batch_size=batch_size, drop_last_batch=drop_last_batch, 
                     remove_columns=remove_columns, keep_in_memory=keep_in_memory, 
                     load_from_cache_file=load_from_cache_file, cache_file_name=cache_file_name,
                     writer_batch_size=writer_batch_size, features=features,
                     disable_nullable=disable_nullable, fn_kwargs=fn_kwargs,
                     num_proc=num_proc, suffix_template=suffix_template,
                     new_fingerprint=new_fingerprint)
      features_map= copy.deepcopy(self.features_map)
      for column in remove_columns if remove_columns is not None else []:
          if column in features_map:
              del features_map[column]
      if self.features_map and "id" not in self.features:
        raise RuntimeError(f"Datstore returned from map must have an {id} column to link to views.")
      return Datastore.from_dataset(ret, features_map=features_map)

    def map_tmp(
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
        parallel_backend_type: Optional[List[str]] = ["dask"]
    ) -> "Datastore":

        if True:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
            with parallel_backend(**parallel_backend_type): # 'distributed', scheduler_host='HOST:PORT'):
                os.environ = prev_env
                shards = [
                    self.shard(num_shards=num_proc, index=rank, contiguous=True, keep_in_memory=keep_in_memory)
                    for rank in range(num_proc)
                ]
                kwds_per_shard = [
                    dict(
                        self=shards[rank],
                        function=function,
                        with_indices=with_indices,
                        input_columns=input_columns,
                        batched=batched,
                        batch_size=batch_size,
                        drop_last_batch=drop_last_batch,
                        remove_columns=remove_columns,
                        keep_in_memory=keep_in_memory,
                        load_from_cache_file=load_from_cache_file,
                        cache_file_name=format_cache_file_name(cache_file_name, rank)
                        if cache_file_name is not None
                        else None,
                        writer_batch_size=writer_batch_size,
                        features=features.copy() if features is not None else None,
                        disable_nullable=disable_nullable,
                        fn_kwargs=fn_kwargs,
                        rank=rank,
                        offset=sum(len(s) for s in shards[:rank]),
                        desc=desc,
                    )
                    for rank in range(num_proc)
                ]
                logger.info("Spawning {} processes".format(num_proc))
                # TODO: do smart jobs allocaiton based on which node the shard of the data is stored
                transformed_shards = Parrallel(n_jobs = num_proc, verbose=1)(delayed(self.__class__._map_single)(self, **kwds) for kwds in kwds_per_shard)
                logger.info("Concatenating {} shards from multiprocessing".format(num_proc))
                result = concatenate_datasets(transformed_shards)
                if new_fingerprint is not None:
                    result._fingerprint = new_fingerprint
                ret = result
        features_map= copy.deepcopy(self.features_map)
        for column in remove_columns if remove_columns is not None else []:
          if column in features_map:
              del features_map[column]
        return Datastore.from_dataset(ret, features_map=features_map)


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
            if  self.features[feature] != features[feature]:
                raise NotImplementedError()
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
        ret = super().remove_columns(column_names=column_names, new_fingerprint=new_fingerprint)
        features_map= copy.deepcopy(self.features_map)
        for column in [column_names] if type(column_names) is str else column_names:
            if column in features_map:
                del features_map[column]
        return Datastore.from_dataset(ret, features_map=features_map)

    @fingerprint_transform(inplace=False)
    def rename_column(self, original_column_name: str, new_column_name: str, new_fingerprint) -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        ret = super().rename_column(original_column_name=original_column_name, new_column_name=new_column_name, new_fingerprint=new_fingerprint)
        features_map= copy.deepcopy(self.features_map)
        if original_column_name in features_map:
            val = features_map[original_column_name]
            del features_map[original_column_name]
            features_map[new_column_name] = val
        return Datastore.from_dataset(ret, features_map=features_map)


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
    def select_new(
        self,
        indices: Iterable,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        new_fingerprint: Optional[str] = None,
    ) -> "Datastore":
        ret = super().select(
            indices=indices,
            keep_in_memory=keep_in_memory,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
            ) 
        if not hasattr(self, 'features_map'): self.features_map = {}
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
    ) -> "DatastoreDict":
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

    # shard doesn't seem to work properly because of pickling problems? Maybe it's because it's being run in Colab with autoload??
    def shard_new(
        self,
        num_shards: int,
        index: int,
        contiguous: bool = False,
        keep_in_memory: bool = False,
        indices_cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
    ) -> "Datastore":
        if not hasattr(self, 'features_map'): self.features_map = {}
        ret = super().shard(num_shards=num_shards,
          index=index,
          contiguous=contiguous,
          keep_in_memory=keep_in_memory,
          indices_cache_file_name=indices_cache_file_name,
          writer_batch_size=writer_batch_size)
        return ret # Datastore.from_dataset(ret, features_map=self.features_map)


    # TODO: Fiix load_from_idsk and save_to_disk to work with current version of datasets

    @staticmethod
    def load_from_disk(dataset_path: str, fs=None, shared_dir=None) -> "Datastore":
      # TODO, move from shared drive to cached drive
        ret = Dataset.load_from_disk(dataset_path=dataset_path, fs=fs)
        dataset_path = os.path.dirname(ret._data_files[0]["filename"])
        with open(
            Path(dataset_path, "state.json").as_posix(), "r", encoding="utf-8"
        ) as state_file:
            state = json.load(state_file)
        ret.features_map =  state.get("features_map")
        for key, values in list(ret.features_map.items()):
            mmap_path = os.path.abspath(os.path.join(dataset_path, values[0]))
            ret.features_map[key][0] =  mmap_path
        return Datastore.from_dataset(ret)
        #todo, do periodic sync with the shared drive, and lazy loading of shareds from shared drive

    def save_to_disk(self, dataset_path: str, move_files=True):
        """
        Save the datastore along with all mmaps and uris in a directory

        Args:
            dataset_path (``str``): path of the dataset directory where the dataset will be saved to
        """
        assert (
            not self.list_indexes()
        ), "please remove all the indexes using `dataset.drop_index` before saving a dataset"
        orig_self = self
        if not move_files:
            self = pickle.loads(pickle.dumps(self))
        os.makedirs(dataset_path, exist_ok=True)
        orig_dataset_path = os.path.dirname(self._data_files[0]["filename"])
        # Write indices if needed
        if self._indices is not None:
            if not self._indices_data_files:
                cache_file_name = os.path.join(dataset_path, "indices.arrow")
                writer = ArrowWriter(path=cache_file_name)
                writer.write_table(self._indices)
                writer.finalize()
                self._indices_data_files = [{"filename": cache_file_name}]
        # Write dataset if needed
        if not self._data_files or any(len(h["transforms"]) > 0 for h in self._inplace_history):
            cache_file_name = os.path.join(dataset_path, "dataset.arrow")
            writer = ArrowWriter(path=cache_file_name)
            writer.write_table(self._data)
            writer.finalize()
            self._data_files = [{"filename": cache_file_name}]
            self._inplace_history = [{"transforms": []}]
        # Copy all files into the dataset directory
        for data_file in self._data_files + self._indices_data_files :
            # Copy file to destination directory
            src = data_file["filename"]
            filename = Path(src).name
            dest = os.path.join(dataset_path, filename)
            if src != dest:
                shutil.move(src, dest)
            # Change path to relative path from inside the destination directory
            data_file["filename"] = filename
        for key, value in list(self.features_map.items()):
            # Copy file to destination directory
            src = value[0]
            filename = Path(src).name
            dest = os.path.join(dataset_path, filename)
            # if the src is not under the 
            if src != dest and os.path.exists(src):
                if filename.startswith(orig_dataset_path):
                  shutil.move(src, dest)
                else:
                  shutil.copy(src, dest)
            # Change path to relative path from inside the destination directory
            self.features_map[key] = [filename]  + value[1:]
        if not move_files:
          return orig_self
        # Get state
        state = self.__getstate__()
        dataset_info = json.loads(state.pop("_info"))
        assert state.get("_data") is None, "arrow table needs to be memory mapped"
        assert state.get("_indices") is None, "arrow table needs to be memory mapped"
        assert all(
            len(h["transforms"]) == 0 for h in state.get("_inplace_history", [])
        ), "in-place history needs to be empty"
        # Serialize state
        with open(os.path.join(dataset_path, "state.json"), "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)
        with open(os.path.join(dataset_path, "dataset_info.json"), "w", encoding="utf-8") as dataset_info_file:
            json.dump(dataset_info, dataset_info_file, indent=2, sort_keys=True)
#        logger.info("Dataset saved in {}".format(dataset_path))
        for key, values in list(self.features_map.items()):
            mmap_path = os.path.abspath(os.path.join(dataset_path, values[0]))
            self.features_map[key][0] =  mmap_path
        return self

    
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


# TODO, convert this function to view sharded datasets across Dask nodes as a single dataset
def concatenate_datasets_shards(
    dsets: List[Dataset],
    info: Optional[Any] = None,
    split: Optional[Any] = None,
    axis: int = 0,
):
    """
    Converts a list of :class:`Dataset` with the same schema into a single :class:`Dataset`.
    Args:
        dsets (:obj:`List[datasets.Dataset]`): List of Datasets to concatenate.
        info (:class:`DatasetInfo`, optional): Dataset information, like description, citation, etc.
        split (:class:`NamedSplit`, optional): Name of the dataset split.
        axis (``{0, 1}``, default ``0``, meaning over rows):
            Axis to concatenate over, where ``0`` means over rows (vertically) and ``1`` means over columns
            (horizontally).
            .. versionadded:: 1.6.0
    """
    if axis == 0 and not all([dset.features.type == dsets[0].features.type for dset in dsets]):
        raise ValueError("Features must match for all datasets")
    elif axis == 1 and not all([dset.num_rows == dsets[0].num_rows for dset in dsets]):
        raise ValueError("Number of rows must match for all datasets")

    # Find common format or reset format
    format = dsets[0].format
    if any(dset.format != format for dset in dsets):
        format = {}
        logger.info("Some of the datasets have disparate format. Resetting the format of the concatenated dataset.")

    # Concatenate tables
    table = concat_tables([dset._data for dset in dsets if len(dset._data) > 0], axis=axis)
    if axis == 1:
        table = update_metadata_with_features(table, None)

    def apply_offset_to_indices_table(table, offset):
        if offset == 0:
            return table
        else:
            array = table["indices"]
            new_array = pc.add(array, pa.scalar(offset, type=pa.uint64()))
            return InMemoryTable.from_arrays([new_array], names=["indices"])

    # Concatenate indices if they exist
    if any(dset._indices is not None for dset in dsets):

        # Datasets with no indices tables are replaced with a dataset with an indices table in memory.
        # Applying an offset to an indices table also brings the table in memory.
        for i in range(len(dsets)):
            if dsets[i]._indices is None:
                dsets[i] = dsets[i].select(range(len(dsets[i])))
        assert all(dset._indices is not None for dset in dsets), "each dataset should have an indices table"

        # An offset needs to be applied to the indices before concatenating
        indices_tables = []
        offset = 0
        for dset in dsets:
            indices_tables.append(apply_offset_to_indices_table(dset._indices, offset))
            offset += len(dset._data)

        # Concatenate indices
        indices_tables = [t for t in indices_tables if len(t) > 0]
        if indices_tables:
            indices_table = concat_tables(indices_tables)
        else:
            indices_table = InMemoryTable.from_batches([], schema=pa.schema({"indices": pa.int64()}))
    else:
        indices_table = None

    # Concatenate infos
    if info is None:
        info = DatasetInfo.from_merge([dset.info for dset in dsets])
    fingerprint = update_fingerprint(
        "".join(dset._fingerprint for dset in dsets), concatenate_datasets, {"info": info, "split": split}
    )

    # Make final concatenated dataset
    concatenated_dataset = Dataset(
        table,
        info=info,
        split=split,
        indices_table=indices_table,
        fingerprint=fingerprint,
    )
    concatenated_dataset.set_format(**format)
    return concatenated_dataset

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
      datastore = Datastore.from_dict({'id':range(10000), 'len':[5]*10000})
      datastore = datastore.add_igzip("txt", "wikitext-2/wiki.train.tokens.gz")
      print (datastore)
      print (datastore[0])
      print (datastore[10100])
    if "-test_memmap" in args:
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
    if "-test_sql" in args:
       datastore = Datastore.from_dataset(load_dataset("oscar", "unshuffled_deduplicated_sw")['train'])
       datastore= datastore.add_sql('text2')
       datastore = Datastore.from_dataset(load_dataset("oscar", "unshuffled_deduplicated_yo")['train'])
       datastore= datastore.move_to_sql('text','text2')
      

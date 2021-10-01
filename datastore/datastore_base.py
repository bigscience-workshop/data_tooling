import subprocess
import tempfile
from threading import Timer, Thread
from multiprocessing import Process
import subprocess
import requests
import multiprocessing
from filelock import UnixFileLock, FileLock
from functools import partial

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))


from data_tooling.datastore.utils.utils import *
from data_tooling.datastore.connectors.sql import *
from data_tooling.datastore.connectors.memmap import *

######################################################################################
# Extensions to Huggingface's datasets to create a datastore that
# interconnects to sqlite and memmap
#
# We want to have mutliple types of storage that ideally can be
# transported as a file transfer with an arrow dataset (perhaps a tar
# file?). So if we have <signature>.arrow, we may have
# fts_<signature>.db (for full text indexing sqlite database) and
# <signature>.db (sqlite database), and <siganture>.mmap (mmap file
# reprsenting a tensor)

#NOTE: datasets uses the terms 'features' and 'columns' interchangably.


class Datastore(Dataset): 
    """
    A class that wraps a Huggingface arrow based Dataset to provide
    support for features bound to memmap and sqlalchemy via the
    package datastore.
    
    Also permits full text indexing and searching (via .filter or
    .search) into a sqlite database for any text feature in a
    datastore.
    """
        
    def __repr__(self):
        ret = FeaturesWithViews(self._info.features)
        ret.views_map = {} if not hasattr(self, "views_map") else self.views_map
        return f"Datastore({{\n    features: {ret},\n    num_rows: {self.num_rows}\n}})"
    

    @classmethod
    def from_dataset(cls, dataset, template_datastore=None, views_map=None, primary_id=None, pipelines_manager=None, id2idx_identity=None,):
        self = cls(
            arrow_table=dataset._data,
            indices_table=dataset._indices,
            info=dataset.info.copy(),
            split=dataset.split,
            fingerprint=dataset._fingerprint,
        )
        if template_datastore is None:
          template_datastore = dataset
        self.mmap_access_cnt=0
        
        if  hasattr(dataset, "id2idx_identity"):
          self.id2idx_identity = dataset.id2idx_identity
        elif  id2idx_identity is not None:
          self.id2idx_identity = id2idx_identity
        elif hasattr(template_datastore, "id2idx_identity"):
          self.id2idx_identity = template_datastore.id2idx_identity
        else:
          self.id2idx_identity = True

        if  hasattr(dataset, "_primary_id"):
          self._primary_id = dataset._primary_id
        elif  primary_id is not None:
          self._primary_id = primary_id
        elif hasattr(template_datastore, "_primary_id"):
          self._primary_id = template_datastore._primary_id
        else:
          self._primary_id = "id"

        if  hasattr(dataset, "views_map"):
          self.views_map = copy.deepcopy(dataset.views_map)
        elif  views_map is not None:
          self.views_map = copy.deepcopy(views_map)
        elif hasattr(template_datastore, "views_map"):
          self.views_map = copy.deepcopy(template_datastore.views_map)
        else:
          self.views_map = {}

        return self

    def _get_mmap(self, path,  dtype, shape):
      shape[0] = max(shape[0], len(self))
      # what happens when the datastore shrinks??
      ret = np_mmap(path, dtype, shape)
      if  not hasattr(self, "mmap_access_cnt"): self.mmap_access_cnt=0
      if self.mmap_access_cnt % 100==0: #let's flush intermittently just in case the OS needs to synch.
        ret.flush()
        self.mmap_access_cnt=0
      self.mmap_access_cnt+=1
      return ret

    # we use class variables to cache sql connections because we don't
    # want it serialized in this instance. TODO: this might take up
    # too much memory, so we might use an LRU cache instead.
    db_table = {}
    def _get_db_table(self, feature):
        if feature in self.views_map and self.views_map[feature]['type'] == 'sql':
          table_name, connection_url = val['table_name'], val['connection_uri']
          if (table_name, connection_uri) in Datastore.db_table:
              table =  Datastore.db_table[(table_name, connection_uri)]
          else:
              if connection_uri in Datastore.db_connection:
                  db =  Datastore.db_connection[connection_uri]
              else:
                  Datastore.db_connection[connection_uri] = db = DatabaseExt(connection_uri)
              Datastore.db_table[(table_name, connection_uri)] = table = db[table_name]
          return table
        else:
          raise RuntimeError(f"{feature} is not a sql type")

    @staticmethod
    def _add_idx(batch, indices, primary_id,):
        batch[primary_id] = indices # will this be shuffled if we are in shuffled mode?
        return batch

    @staticmethod
    def _move_to_mmap_col(batch, src_feature, primary_id, path, dtype, shape):
        ret = np_mmap(path, dtype, shape)
        ret[batch[primary_id]] = batch[src_feature]

    @classmethod
    def from_mmap(cls,  feature_view, shape, path=None, dtype='float32', dtype_str_len=100, primary_id="id", batch_size=1000, num_proc=4):
      return cls.from_dict({}).add_mmap(feature_view=feature_view, shape=shape, path=path, dtype=dtype, dtype_str_len=dtype_str_len, primary_id=primary_id, batch_size=batch_size, num_proc=num_proc)


    def move_to_mmap(self, src_feature, dst_feature_view=None, shape=None, path=None, dtype='float32', dtype_str_len=100, primary_id="id", batch_size=1000, num_proc=4):
      if dst_feature_view in (src_feature, None):
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
      shape[0] = max(shape[0],len(self))
      self.add_mmap(feature_view=dst_feature_view, shape=shape, path=path, dtype=dtype, primary_id=primary_id, batch_size=batch_size, num_proc=num_proc,) #don't pass in the pipelines_manager
      val = self.views_map[dst_feature_view]
      self.map(Datastore._move_to_mmap_col, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'src_feature':src_feature, 'primary_id':primary_id, 'path': val['path'], 'dtype': val['dtype'], 'shape':shape})
      self= self.remove_columns(src_feature)
      if hasattr(self, 'pipelines_manager') and self.pipelines_manager not in (None, pipelines_manager):
          print(f"warning: resetting the metadta_manager to {pipelines_manager}")
      if pipelines_manager is not None:
          self.pipelines_manager = pipelines_manager
      #only apply the pipelines_manager if we are moving to a new feature and that feature is monitored. 
      if pipelines_manager is not None and src_feature.startswith ("__tmp__") and  dst_feature_view in pipelines_manager.features_monitored() and self.pipelines_manager.postprocess:
        self = self.map(pipelines_manager.postprocess,  batch_size=batch_size, batched=True, num_proc=num_proc)
      return self

    def add_mmap(self, feature_view, shape, path=None, dtype='float32', dtype_str_len=100, primary_id="id", batch_size=1000, num_proc=4):
      """"mapping a feature/columun to a memmap array accessed by row"""
      if not hasattr(self, 'views_map'): self.views_map = {}
      if hasattr(self, '_primary_id') and self._primary_id != primary_id:
        raise RuntimeError(f"attempting to reset the index to {primary_id}")
      else:
        self._primary_id = _primary_id
      if hasattr(self, 'pipelines_manager') and self.pipelines_manager not in (None, pipelines_manager):
          print(f"warning: resetting the metadata_manager to {pipelines_manager}")
      if pipelines_manager is not None:
          self.pipelines_manager = pipelines_manager
      if not self.cache_files:
        dataset_path = get_temporary_cache_files_directory()
      else:  
        dataset_path = os.path.dirname(self.cache_files[0]['filename'])
      if path is None:
          path = os.path.abspath(os.path.join(dataset_path, feature_view+".mmap"))
      shape = list(shape)
      shape[0] = max(shape[0], len(self))
      if primary_id not in self.features:
        if len(self) == 0 and shape[0] > 0:
            self = Datastore.from_dataset(Dataset.from_dict({primary_id: range(shape[0])}), self)
            ids = dict([(a,1) for a in range(len(self))])
            self.id2idx_identity = True
        else:
            self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'primary_id': primary_id})
            ids = dict([(a,1) for a in range(len(self))])
            self.id2idx_identity = True
      else:
        ids = dict([(a,1) for a in self[primary_id]])
      missing_ids = []
      for id in range(shape[0]):
          if id not in ids:
            missing_ids.append(id)
      if missing_ids:
            self = self.add_batch({primary_id: missing_ids})
            if not hasattr(self, 'id2idx_identity'):  self.id2idx_identity = True
            if self.id2idx_identity:
              contiguous, start, end = is_contiguous(missing_ids)
              self.id2idx_identity = start ==len(self) and contiguous
            else:
              self.id2idx_identity = False
      if not isinstance(dtype, str):
          dtype =np.dtype(dtype).name

      self.views_map[feature_view] = {'type':"mmap", 'path': path,  'dtype': dtype, 'shape': shape}
      if pipelines_manager is not None and feature_view in pipelines_manager.features_monitored() and self.pipelines_manager.postprocess:
        self = self.map(pipelines_manager.postprocess,  batch_size=batch_size, batched=True, num_proc=num_proc)
      return self



    def move_to_sql(self, src_feature_to_dst_views_map, table_name=None, connection_uri=None,  primary_id="id",  batch_size=1000, num_proc=4, fts_connection_uri=None):
      if table_name is None:
          #print (self.info.builder_name, self.info.config_name)
          table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}"
      if not connection_uri:
          connection_uri="sqlite:///"+self.cache_files[0]['filename'].replace(".arrow", ".db")
      table = Datastore._get_db_table(self, table_name, connection_uri)
      if type(src_feature_to_dst_views_map) is list:
        src_feature_to_dst_views_map = dict(src_feature_to_dst_views_map)
      elif type(src_feature_to_dst_views_map) is str:
        src_feature_to_dst_views_map = {src_feature_to_dst_views_map: src_feature_to_dst_views_map}
      feature_view = []
      for src_feature, dst_feature_view in list(src_feature_to_dst_views_map.items()):
        if src_feature == dst_feature_view:
          self = self.rename_column(src_feature, "__tmp__"+src_feature)
          src_feature_to_dst_views_map["__tmp__"+src_feature] = dst_feature_view
          del src_feature_to_dst_views_map[src_feature]
          src_feature = "__tmp__"+src_feature
        value = self[0][src_feature]
        if type(value) is str: #we don't want to save as json type just in case
            value="**"
        dtype = table.db.types.guess(value)
        feature_view.append((dst_feature_view, dtype))
      self.add_sql(feature_view=feature_view, table_name=table_name, connection_uri=connection_uri, primary_id=primary_id, batch_size=batch_size, num_proc=num_proc, fts_connection_uri=fts_connection_uri)
      self = self.map(Datastore.upsert_sql_from_batch, batch_size=batch_size, batched=True, num_proc=1 if connection_uri=="sqlite://" else num_proc, fn_kwargs={'views_map':self.views_map, 'primary_id':primary_id, 'src_feature_to_dst_views_map': src_feature_to_dst_views_map})
      self = self.remove_columns(src_feature)
      return self

    @classmethod
    def from_sql(cls,  feature_view, table_name, connection_uri, dtype="str", primary_id="id",  batch_size=1000, num_proc=4,  fts_connection_uri=None):
      return cls.from_dict({}).add_sql(feature_view=feature_view, table_name=table_name, connection_uri=connection_uri, dtype=dtype, primary_id=primary_id, batch_size=batch_size, num_proc=num_proc,  fts_connection_uri=fts_connection_uri)

    def add_sql(self, feature_view=None, table_name=None, connection_uri=None, dtype="str", primary_id="id",  batch_size=1000, num_proc=4):
        """
        mapping one or more columns/features to a sql database. creates a sqlalchmey/dataset dynamically with primary_id as the primary key. 
        TODO: remember to strip passwords from any connection_uri. passwords should be passed as vargs and added to the conneciton url dynamically
        passwords should not be perisisted.
        NOTE: this dataset will not automatically change if the database changes, and vice versa. periodically call this method again to sync the two or create callbacks/triggers in your code.
        """
        if not hasattr(self, 'views_map'): self.views_map = {}
        if hasattr(self, '_primary_id') and self._primary_id != primary_id:
          raise RuntimeError(f"attempting to reset the index to {primary_id}")
        else:
          self._primary_id = primary_id
        if table_name is None:
          #print (self.info.builder_name, self.info.config_name)
          table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}"
          #print (table_name)
        if not connection_uri:
          connection_uri="sqlite:///"+self.cache_files[0]['filename'].replace(".arrow", ".db")
        if not fts_connection_uri:
          if connection_uri.starts("sqlite:"):
            fts_connection_uri = connection_uri
          else:
            fts_connection_uri="sqlite:///"+self.cache_files[0]['filename'].replace(".arrow", ".db")
        if type(feature_view) is str:
          feature_view = [(feature_view, dtype)]
        if type(feature_view) is dict:
          feature_view =  list(feature_view.items())
        table = self._get_db_table(table_name, connection_uri)
        if not feature_view and table.columns:
            feature_view = table.columns
        elif not feature_view:
            raise RuntimeError(f"No feature_view(s) and no column definition for table view {table_name}")
        table_ids = table.find(_columns=primary_id)
        if primary_id not in self.features:
          if len(self) == 0 and table_ids:
            self = Datastore.from_dataset(Dataset.from_dict({primary_id: range(max([id[primary_id] for id in table_ids]))}), self)
            ids = dict([(a,1) for a in range(len(self))])
            self.id2idx_identity = True
          else:
            self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'id': primary_id})
            ids = dict([(a,1) for a in range(len(self))])
            self.id2idx_identity = True
        else:
          ids = dict([(a,1) for a in self[primary_id]])
        missing_ids = []
        for id in table_ids:
          if id[primary_id] not in ids:
            missing_ids.append(id[primary_id])
        if missing_ids:
            self = self.add_batch({primary_id: missing_ids})
            if not hasattr(self, 'id2idx_identity'):  self.id2idx_identity = True
            if self.id2idx_identity:
              contiguous, start, end = is_contiguous(missing_ids)
              self.id2idx_identity = start ==len(self) and contiguous
            else:
              self.id2idx_identity = False
        for col in feature_view:
            if type(col) is tuple:
              col, dtype = col
            else:
              dtype=None
            if col == primary_id:
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
            fts_table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}_{col}_fts_idx"
            self.views_map[col] = {'type':'sql', 'connection_uri': connection_uri, 'table_name': table_name, 'fts_connection_uri': fts_connection_uri, 'fts_table_name': fts_table_name}
        return self
    
    def add_fts(self, feature_view, fts_table_name=None, fts_connection_uri=None,  primary_id="id",  batch_size=1000, num_proc=4):
      if type(feature_view) is str:
        feature_view = [feature_view,]
      for col in feature_view:
        fts_table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}_{col}_fts_idx"
        self.views_map[col] = {'type':'fts_only', 'connection_uri': fts_connection_uri, 'table_name': fts_table_name, 'fts_connection_uri': fts_connection_uri, 'fts_table_name': fts_table_name}


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
        sql_query: Optional[dict] = None,
        fts_query: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        distributed_context: DistributedContext = None, # we only use the next parameters if there is a distributed_context.
        intermediate_sort: Optional[bool] = True, 
        final_reduce: Optional[bool] =True,  
        shared_dir: Optional[str] =None, 
        gzip_output: Optional[bool]=True,
        delete_input_files_on_finalize: Optional[bool] = True,
    ) -> "Datastore":
      """
      the same as datasets.filter except we add sql_query and
      fts_query. sql_query applies a sql query to features/columns
      that are mapped to sql which could be faster than doing a normal
      "filter" function.  the sql_query parameters are the same as the
      "find" method from dataset.Table.  For example:
      dataset.filter(sql_query={'lang':'ru'}) will return those items
      in the dataset that has the language 'ru'.
      """
      if not hasattr(self, 'views_map'): self.views_map = {}
      ret = self
      if sql_query or fts_query:
        if not sql_query:
          sql_query={}
        if not fts_query:
          fts_query={}
        ids2rank = {}
        found_table = None
        sql_query2 = {}
        fts_query2 = {}
        for feature_view, query in sql_query.items():
          val = self.views_map.get(feature_view)
          if not val or val['type']!='sql':
            raise RuntimeError(f"{feature_view} is not a sql or fts view and can not be filtered as such")
          if val['type']=='sql':
            connection_uri, table_name = val['fts_connection_uri'], val['fts_table']
            sql_query2[(connection_uri, table_name)] = sql_query2.get((connection_uri, table_name), [])+[query]
        for feature_view, query in fts_query.items():
          val = self.views_map.get(feature_view)
          if not val or val['type'] not in ('sql', 'fts_only'):
            raise RuntimeError(f"{feature_view} is not a sql, or fts view and can not be filtered as such")
          if val['type']=='sql' and 'fts_connection_uri' in view:
            connection_uri, table_name = val['connection_uri'], val['table_name']
            fts_query2[(connection_uri, table_name)] = sql_query2.get((connection_uri, table_name), [])+[query]
          elif val['type'] == 'fts_only':
            connection_uri, table_name = val['connection_uri'], val['table_name']
            fts_query2[(connection_uri, table_name)] = sql_query2.get((connection_uri, table_name), [])+[query]            

        for key, query in sql_query2.items():
          (connection_uri, table_name) = key
          if key in fts_query2:
            query['_fts_query'] = fts_query2[key]
            del fts_query2[key]
          query['_columns'] = [self._primary_id]
          table = self._get_db_table(table_name, connection_uri)
          for val in table.find(*[], **query):
            ids2rank[val[self._primary_id]] = min(ids2rank.get(val[self._primary_id], (100000+val[self._primary_id]) if 'rank' not in val else val['rank']),  (100000+val[self._primary_id]) if 'rank' not in val else val['rank'])

        for key, query in fts_query2.items():
          (connection_uri, table_name) = key
          query2={}
          query2['_fts_query'] = query
          query2['_columns'] = [self._primary_id]
          table = self._get_db_table(table_name, connection_uri)
          for val in table.find(*[], **query2):
            ids2rank[val[self._primary_id]] = min(ids2rank.get(val[self._primary_id], (100000+val[self._primary_id]) if 'rank' not in val else val['rank']),  (100000+val[self._primary_id]) if 'rank' not in val else val['rank'])

        if ids2rank:
          ids = list(ids2rank.keys())
          ids.sort(key=lambda a: ids2rank[a])
          if hasattr(self, 'id2idx_identity') and self.id2idx_identity:
            ret = self.select(ids)
            ret.id2idx_identity=False
          else:
            if function:
              function = lambda example: example['id'] in ids and function(example) 
            else:
              function = lambda example: example['id'] in ids
      if function is None and remove_columns is None:
        return ret

      # just copy the filter function here, but use Datastore's map function.
      if len(self.list_indexes()) > 0:
            raise DatasetTransformationNotAllowedError(
                "Using `.filter` on a dataset with attached indexes is not allowed. You can first run `.drop_index() to remove your index and then re-add it.`"
            )
      
      if function is None:
          function = lambda x: True  # noqa: E731

      if isinstance(input_columns, str):
          input_columns = [input_columns]

      if input_columns is not None:
          for input_column in input_columns:
              if input_column not in self._data.column_names:
                  raise ValueError(
                      "Input column {} not in the dataset. Current columns in the dataset: {}".format(
                          input_column, self._data.column_names
                      )
                  )

      if fn_kwargs is None:
          fn_kwargs = {}
      fn_kwargs["input_columns"] = input_columns

      # return map function
      return ret.map(
          partial(get_indices_from_mask_function, function=function, with_indices=with_indices),
          batched=True,
          with_indices=with_indices,
          features=self.features,
          batch_size=batch_size,
          remove_columns=remove_columns,
          keep_in_memory=keep_in_memory,
          load_from_cache_file=load_from_cache_file,
          cache_file_name=cache_file_name,
          writer_batch_size=writer_batch_size,
          fn_kwargs=fn_kwargs,
          num_proc=num_proc,
          suffix_template=suffix_template,
          new_fingerprint=new_fingerprint,
          distributed_context=distributed_context,
          intermediate_sort=intermediate_sort,
          final_reduce=final_reduce,
          shared_dir=shared_dir,
          delete_input_files_on_finalize=delete_input_files_on_finalize,
      )

    # note that while the primary_id corresponds to an item in an
    # external storage, accessing an arrow dataset by datataset[index]
    # will not be guranteed to get the corresponding id. a[0] will
    # return the first item in the current subset of the dataset.  but
    # a[0] does not necessarily return {'id':0, ...}  instead, a[0]
    # might return {'id': 10, 'mmap_embed': <array correponding to the
    # 10th location in the mmap file>}.  To get dataset items by 'id',
    # use either filter or check the property id2idx_identity to
    # determine if the id corresponds to the index of the table.
    def _getitem(
        self,
        key: Union[int, slice, str], # should this be list as well??
        format_type=None,
        format_columns=None,
        output_all_columns=False,
        format_kwargs=None,
    ) -> Union[Dict, List]:
        if not hasattr(self, 'views_map'): self.views_map = {}
        # assumine we do error checking re format_columns and output_all_columns at a higher level??
        format_columns = copy.copy(format_columns)
        # this is the case where we are not getting any views.
        if (not self.views_map) or (type(key) is str and key not in self.views_map):
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
          if key in self.views_map:
            key = self.primary_id
        missing=[]
        if format_columns:
            for c in copy.copy(format_columns):
                if c in self.views_map:
                     missing.append(c)
                     format_columns.remove(c)
            if self.primary_id not in format_columns:
                format_columns.append(self.primary_id)
            else:
                missing.append(self.primary_id)

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
        if type(orig_key) is str and format_columns and self.primary_id in format_columns:
            format_columns.remove(self.primary_id)
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
                items = list(self.views_map.items())
            else:
                items = [(column, self.views_map[column]) for column in format_columns if column in self.views_map]
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
                elif val['type'] == 'sql':
                    sql_results[(val['table_name'], val['connection_uri'])] = sql_results.get((val['table_name'], val['connection_uri']),[])+[feature]
            for table_connection, features in sql_results.items():
                table_name, connection_uri = table_connection
                table= self._get_db_table(table_name, connection_uri)
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
        if format_type in ("custom", "torch", "tensorflow", None) and type(outputs_or_keys)  not in (da.DataFrame, pd.DataFrame): 
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
                keys = outputs_or_keys[self.primary_id]
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
            if self.primary_id in outputs and format_columns and self.primary_id not in format_columns: del outputs[self.primary_id] 
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
        elif format_type in ("dask", "pandas") or type(outputs_or_keys) in (da.DataFrame, pd.DataFrame):
            # do we do transforms for this case??
            df = pd
            if format_type in ("dask",) or type(outputs_or_keys) in (da.DataFrame,):
              df = dd
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
            elif isinstance(outputs_or_keys, dict) or isinstance(outputs_or_keys, df.DataFrame):
                outputs = outputs_or_keys
                outputs =df.DataFrame(outputs)
                keys = outputs_or_keys[self.primary_id]
                contiguous, start, end = is_contiguous(keys)
            else:
                raise RuntimeError("got unknown outputs or keys type")
            if outputs is None:
                outputs = df.DataFrame()
            outputs = getitems(self, outputs,  keys, contiguous, start, end, format_columns, output_all_columns, mmap_by_items=True)
            if self.primary_id in outputs and format_columns and self.primary_id not in format_columns: 
              outputs.drop(self.primary_id, axis=1) 
            if len(format_columns) == 1:
              outputs = outputs[format_columns[0]]
            return outputs
        raise RuntimeError("got unknown outputs or keys type")

    @staticmethod
    def upsert_sql_from_batch(batch, views_map, primary_id, src_feature_to_dst_views_map):
      sql_results={}
      for src_feature, dst_feature in src_feature_to_dst_views_map.items() if src_feature_to_dst_views_map is not None else zip(batch.keys(),batch.keys()):
        if views_map.get(dst_feature):
          val = views_map[dst_feature]
          if val['type'] == 'sql':
            sql_results[(val['table_name'], val['connection_uri'])] = sql_results.get((val['table_name'], val['connection_uri']),[])+[(src_feature, dst_feature)]
      for key, features in sql_results.items():
        table_name, connection_uri = key
        db = DatabaseExt(connection_uri)
        with db:
            table = db[table_name]
            batch2 = []
            for i in range(len(batch[primary_id])):
              batch2.append(dict([(feature[1], batch[feature[0]][i]) for feature in features+[(primary_id,primary_id)]]))               
            try:
              table.insert_many(batch2)
            except:
              batch2 = []
              for i in range(len(batch[primary_id])):
                batch2.append(dict([(feature[1], batch[feature[0]][i]) for feature in features+[(primary_id,primary_id)]]))    
              table.update_many(batch2, [primary_id])
            batch2 = None

    PERSIST_IN_ARROW = 0
    STATIC_VIEWS = 1
    UPDATE_VIEWS = 2

    @staticmethod
    def map_fn_with_indices_and_handle_views(batch, indices, map_fn, fn_kwargs, handle_views, views_map, primary_id):
      ret = map_fn(batch, indices, **fn_kwargs)
      if ret is not None and views_map:
        if views_map and primary_id not in ret:
          raise RuntimeError(f"Datstore returned from map must have an {primary_id} column to link to views.")
        if handle_views != DataStore.PERSIST_IN_ARROW:
          for key in views_map:
            if handle_views == Datastore.UPDATE_VIEWS:
              if val['type'] == 'mmap':
                  mmap_array = np_mmap(val['path'], val['dtype'], val['shape'])
                  mmap_array[batch[primary_id]] = batch[feature]                     
            elif handle_views == Datastore.STATIC_VIEWS:
              if key in ret:
                del ret[key]
          if handle_views == 2: Datastore.upsert_sql_from_batch(ret, views_map, primary_id, None)
      return ret

    @staticmethod
    def map_fn_and_handle_views(batch, map_fn, fn_kwargs, handle_views, views_map, primary_id):
      ret = map_fn(batch, **fn_kwargs)
      if ret is not None and views_map:
        if views_map and primary_id not in ret:
          raise RuntimeError(f"Datstore returned from map must have an {primary_id} column to link to views.")
        if handle_views != Datastore.PERSIST_IN_ARROW:
          for key in views_map:
            if handle_views == Datastore.UPDATE_VIEWS:
              if val['type'] == 'mmap':
                  mmap_array = np_mmap(val['path'], val['dtype'], val['shape'])
                  mmap_array[batch[primary_id]] = batch[feature]                     
            elif handle_views == Datatsore.STATIC_VIEWS:
              if key in ret:
                del ret[key]
          if handle_views == 2: Datastore.upsert_sql_from_batch(ret, views_map, primary_id, None)
      return ret

    #:arg handle_views: tells us how to handle views. 
    #PERSIST_IN_ARROW - all data returned will be persisted to arrow storage and not views. this will detach all views.
    #STATIC_VIEWS - keep the views attached to external storage without change. *default*
    #UPDATE_VIEWS - update views based on what is returned by the map function. this will create a side-effect.
    #Updating views might create an unepxected side-effect on caching.  Use caching with cuation when editing views.
    def map(self, 
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = False,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[Union[str, List[str]]] = None,
        keep_in_memory: bool = False,
        load_from_cache_file: bool = None,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = False,
        fn_kwargs: Optional[dict] = None,
        num_proc: Optional[int] = None,
        suffix_template: str = "_{rank:05d}_of_{num_proc:05d}",
        new_fingerprint: Optional[str] = None,
        desc: Optional[str] = None,
        #add_memmap_views=None,
        #add_sql_views=None,
    ) -> "Datastore":
      if not hasattr(self, 'views_map'): self.views_map = {}
      views_map= copy.deepcopy(self.views_map)
      for column in remove_columns if remove_columns is not None else []:
          if column in views_map:
              del views_map[column]
      if handle_views != Datastore.PERSIST_IN_ARROW:
        fn_kwargs = {'fn_kwargs': fn_kwargs, 'views_map': views_map, 'map_fn': function, 'handle_views': handle_views, 'primary_id': self.primary_id}
        if with_indices:
            function = Datastore.map_fn_with_indices_and_handle_views
        else:
            function = Datastore.map_fn_and_handle_views
      if shared_dir is None:
        shared_dir = self.shared_dir
      ret= super().map(function=function, with_indices=with_indices, input_columns=input_columns,
                     batched=batched, batch_size=batch_size, drop_last_batch=drop_last_batch, 
                     remove_columns=remove_columns, keep_in_memory=keep_in_memory, 
                     load_from_cache_file=load_from_cache_file, cache_file_name=cache_file_name,
                     writer_batch_size=writer_batch_size, features=features,
                     disable_nullable=disable_nullable, fn_kwargs=fn_kwargs,
                     num_proc=num_proc, suffix_template=suffix_template,
                     new_fingerprint=new_fingerprint, desc=desc,)
      for column in remove_columns if remove_columns is not None else []:
         if column in self.views_map and column in ret:
           print (f"warning: the map function returned a column {column} which is the same as a detached view. this column will be persisted to arrow.")
      return ret

    @transmit_format
    @fingerprint_transform(inplace=False)
    def add_column(self, name: str, column: Union[list, np.array], new_fingerprint: str):
        if not hasattr(self, 'views_map'): self.views_map = {}
        if name in self.views_map:
            raise RuntimeError(f"column {name} is alredy a view")
        ret = super().add_column(name=name, column=column, new_fingerprint=new_fingerprint)
        return Datastore.from_dataset(ret, self)

    def class_encode_column(self, column: str) -> "Datastore":
        if not hasattr(self, 'views_map'): self.views_map = {}
        if column in self.views_map:
            raise NotImplementedError()
        ret = super().class_encode_column(column)
        return Datastore.from_dataset(ret, self)
    
    @fingerprint_transform(inplace=False)
    def flatten(self, new_fingerprint, max_depth=16) -> "Datastore":
        if not hasattr(self, 'views_map'): self.views_map = {}
        ret = super().flatten(new_fingerprint, max_depth)
        return Datastore.from_dataset(ret, self)

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
        if not hasattr(self, 'views_map'): self.views_map = {}
        for feature in self.views_map:
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
        return Datastore.from_dataset(ret, self)


    #renaming a column view mapped to a sql database will not change the name in the database.
    @fingerprint_transform(inplace=False)
    def rename_column(self, original_column_name: str, new_column_name: str, new_fingerprint) -> "Datastore":
        if not hasattr(self, 'views_map'): self.views_map = {}
        views_map= copy.deepcopy(self.views_map)
        if original_column_name in views_map:
            val = views_map[original_column_name]
            del views_map[original_column_name]
            views_map[new_column_name] = val
            return Datastore.from_dataset(self, self, views_map=views_map)
        ret = super().rename_column(original_column_name=original_column_name, new_column_name=new_column_name, new_fingerprint=new_fingerprint)
        return Datastore.from_dataset(ret, self, views_map=views_map)
        
    #renaming a column view mapped to a sql database will not change the name in the database.
    @fingerprint_transform(inplace=False)
    def rename_columns(self, column_mapping: Dict[str, str], new_fingerprint)  -> "Datastore":
        if not hasattr(self, 'views_map'): self.views_map = {}
        views_map= copy.deepcopy(self.views_map)
        for original_column_name, new_column_name in list(column_mapping.items()):
            val = views_map[original_column_name]
            del views_map[original_column_name]
            views_map[new_column_name] = val
            del column_mapping[original_column_name]
        if not column_mapping:
          return Datastore.from_dataset(self, self, views_map=views_map) 
        ret = super().rename_column(column_mapping=column_mapping, new_fingerprint=new_fingerprint)
        return Datastore.from_dataset(ret, self, views_map=views_map)

    def prepare_for_task(self, task: Union[str, TaskTemplate]) -> "Datastore":
        if not hasattr(self, 'views_map'): self.views_map = {}
        ret = super().prepare_for_task(task)
        return Datastore.from_dataset(ret, self)


    @transmit_format
    @fingerprint_transform(inplace=False, ignore_kwargs=["cache_file_name"])
    def flatten_indices(
        self,
        keep_in_memory: bool = False,
        cache_file_name: Optional[str] = None,
        writer_batch_size: Optional[int] = 1000,
        features: Optional[Features] = None,
        disable_nullable: bool = True,
        new_fingerprint: Optional[str] = None,
    ) ->  "Datastore":
        if not hasattr(self, 'views_map'): self.views_map = {}
        ret = super().flatten_indices(
            keep_in_memory=keep_in_memory,
            cache_file_name=cache_file_name,
            writer_batch_size=writer_batch_size,
            features=features,
            disable_nullable=disable_nullable,
            new_fingerprint=new_fingerprint,
            )
        return Datastore.from_dataset(ret, self)

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
        if not hasattr(self, 'views_map'): self.views_map = {}
        if column in self.views_map:
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
        return Datastore.from_dataset(ret, self)


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
        if not hasattr(self, 'views_map'): self.views_map = {}
        ret = super().shuffle(
            seed=seed,
            generator=generator,
            keep_in_memory=keep_in_memory,
            load_from_cache_file=load_from_cache_file,
            indices_cache_file_name=indices_cache_file_name,
            writer_batch_size=writer_batch_size,
            new_fingerprint=new_fingerprint,
            )
        return Datastore.from_dataset(ret, self)
  
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
        if not hasattr(self, 'views_map'): self.views_map = {}
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
            ret[key] = Datastore.from_dataset(ret, self)
        return ret


    @transmit_format
    @fingerprint_transform(inplace=False)
    def add_item(self, item: dict, new_fingerprint: str):
        if not hasattr(self, 'views_map'): self.views_map = {}
        ret = super().add_item(item=item,
          new_fingerprint=new_fingerprint)
        return Datastore.from_dataset(ret, self)

    @transmit_format
    @fingerprint_transform(inplace=False)
    def add_batch(self, batch, new_fingerprint: str):
        """Add batch  to Dataset.

        Args:
            batch (Datastore of same schema or dict): batch data to be added.

        Returns:
            :class:`Datastore`
        """
        if not hasattr(self, 'views_map'): self.views_map = {}
        # take care of the case where views_map needs to be merged and the batch's indices are 
        # offsetted
        if type(batch) is dict:
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
              print (item_table._data)
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
        return Datastore.from_dataset(ret, self)

    def align_labels_with_mapping(self, label2id: Dict, label_column: str) -> "Datastore":
        if not hasattr(self, 'views_map'): self.views_map = {}
        ret = super().align_labels_with_mapping(label2id=label2id,
            label_column=label_column)
        return Datastore.from_dataset(ret, self)
        
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
        from dataset.io.parquet import ParquetDatasetReader

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


    def save_to_disk(self, dataset_path: str, fs=None, move_files=False):
      # move_files means delete the old files as we create the new files in dataset_path.
        """
        Saves a dataset to a dataset directory, or in a filesystem using either :class:`~filesystems.S3FileSystem` or
        any implementation of ``fsspec.spec.AbstractFileSystem``.
        Note regarding sliced datasets:
        If you sliced the dataset in some way (using shard, train_test_split or select for example), then an indices mapping
        is added to avoid having to rewrite a new arrow Table (save time + disk/memory usage).
        It maps the indices used by __getitem__ to the right rows of the arrow Table.
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
        # Save views data, dataset + indices + state + info
        fs.makedirs(dataset_path, exist_ok=True)
        views_map_copy = copy.deepcopy(self.views_map)
        for key, value in list(self.views_map.items()):
            # Copy or move file to destination directory
            if 'connection_uri' in value:
              if "sqlite:///" in value['connection_uri']:
                src = value['connection_uri'].replace("sqlite:///", "")
                if value['connection_uri'] in Datastore.db_connection:
                  db = Datastore.db_connection[value['connection_uri']]
                  db.close()
                  del  Datastore.db_connection[value['connection_uri']]
                  db = None
                for key in list(Datastore.db_table.keys()):
                  if key[1] == value['connection_uri']:
                    del Datastore.db_table[key]
                value['connection_uri'] = "sqlite:///"+Path(src).name
              else:
                continue
            else:
              src = value['path']
              value['path'] = Path(src).name
            filename = Path(src).name
            dest = os.path.join(dataset_path, filename)

            # if the src is not the same as dest, we want to move or copy
            if src != dest and os.path.exists(src):
                if move_files:
                  shutil.move(src, dest)
                else:
                  shutil.copy(src, dest)

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
                "views_map",
                "id2idx_identity",
                "_primary_id",
            ]
        }
        self.views_map = views_map_copy
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


        with fs.open(Path(dataset_path, config.DATASET_ARROW_FILENAME).as_posix(), "wb") as dataset_file:
            with ArrowWriter(stream=dataset_file) as writer:
                writer.write_table(self._data)
                writer.finalize()

        if self._indices is not None:
            with fs.open(Path(dataset_path, config.DATASET_INDICES_FILENAME).as_posix(), "wb") as indices_file:
                with ArrowWriter(stream=indices_file) as writer:
                    writer.write_table(self._indices)
                    writer.finalize()
        if move_files:
          orig_dataset_path = os.path.dirname(self.cache_files[0]['filename'])
          arrow_file = Path(orig_dataset_path, config.DATASET_ARROW_FILENAME).as_posix()
          if os.path.exists (arrow_file):
            os.unlink(arrow_file)
          indices_file = Path(orig_dataset_path, config.DATASET_INDICES_FILENAME).as_posix()
          if os.path.exists (indices_file):
            os.unlink(indices_file)
          json_file = Path(orig_dataset_path, config.DATASET_STATE_JSON_FILENAME).as_posix()
          if os.path.exists (json_file):
            os.unlink(json_file)
          info_file = Path(orig_dataset_path, config.DATASET_INFO_FILENAME).as_posix()
          if os.path.exists (info_file):
            os.unlink(info_file)
          license_file = Path(orig_dataset_path, config.LICENSE_FILENAME).as_posix()
          if os.path.exists (license_file):
            os.unlink(license_file)
          for cache_filename in self.cache_files:
            if os.path.exists  (cache_filename["filename"]):
              os.unlink(cache_filename["filename"])
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
        if move_files:
          return Datastore.load_from_disk(dataset_path, fs=fs,)
        else:
          return self


    @staticmethod
    def load_from_disk(dataset_path: str, fs=None, keep_in_memory: Optional[bool] = None) -> "Datastore":
        ret = Dataset.load_from_disk(dataset_path=dataset_path, fs=fs, keep_in_memory=keep_in_memory)
        with open(
            Path(dataset_path, config.DATASET_STATE_JSON_FILENAME).as_posix(), "r", encoding="utf-8"
        ) as state_file:
            state = json.load(state_file)
        ret.views_map =  state.get("views_map")
        ret.id2idx_identity =  state.get("id2idx_identity")
        fs = fsspec.filesystem("file") if fs is None else fs
        for key, value in list(ret.views_map.items()):
            if 'connection_uri' in value:
              if "sqlite:///" in value['connection_uri']:
                src = Path(value['connection_uri'].replace("sqlite:///", "")).name
              else:
                continue
            else:
              src = Path(value['path']).name
            if is_remote_filesystem(fs):
                data_path = os.path.join(dataset_path, src)
                src_dataset_path = extract_path_from_uri(data_path) 
                tmp_dir = tempfile.TemporaryDirectory()
                data_path = Path(tmp_dir.name, src_dataset_path)
                fs.download(src_dataset_path, data_path.as_posix(), recursive=True)
            else:
                data_path = os.path.abspath(os.path.join(dataset_path, src))
            if 'connection_uri' in value:
              ret.views_map[key]['connection_uri'] =  "sqlite:///"+data_path
            else:
              ret.views_map[key]['path'] =  data_path
        return Datastore.from_dataset(ret)


class FeaturesWithViews(Features):
    """ an extension of Features that allows printing of the views' name as well """
    def copy(self):
        ret= FeaturesWithViews(super().copy())
        if hasattr(self, "views_map"):
            ret.views_map = copy.deepcopy(self.views_map)
        return ret

    def __repr__(self):
        ret =  "{"+"\n\t\t".join([f"'{a[0]}': {a[1]}" for a in self.items() if a[0] not in self.views_map])
        if self.views_map:
            ret = ret+"\n\t\t"+"\n\t\t".join(f"'{a[0]}': View({a[1]})" for a in  self.views_map.items())
        ret +="\n}"
        return ret


class Sqlite3FTSIndex(BaseIndex):
    """Sqlite3 FTS Index class for indexing"""
    def __init__(
        self,
        table: TableSharded = None,
        column: Optional[str] = None
    ):
        self.table=table
        self.column=column
        
    def search(self, query, k: int = 10) -> SearchResults:
        hits= list(self.table.find(_fts_query=[(self.column, query)], _limit=k))
        return SearchResults([hit["rank"] for hit in hits], [int(hit["rowid"]) for hit in hits]) # this should be a generator or we feed in a row_type of a signle SearchResult
        
    def search_batch(self, queries, k: int = 10) -> BatchedSearchResults:
        """Find the nearest examples indices to the query.
        Args:
            queries (`Union[List[str], np.ndarray]`): The queries as a list of strings if `column` is a text index or as a numpy array if `column` is a vector index.
            k (`int`): The number of examples to retrieve per query.
        Ouput:
            total_scores (`List[List[float]`): The retrieval scores of the retrieved examples per query.
            total_indices (`List[List[int]]`): The indices of the retrieved examples per query.
        """
        total_scores, total_indices = [], []
        for query in queries:
            scores, indices = self.search(query, k)
            total_scores.append(scores)
            total_indices.append(indices)
        return BatchedSearchResults(total_scores, total_indices)

    def save(self, file: Union[str, PurePath]):
        """Serialize the index on disk"""
        raise NotImplementedError

    @classmethod
    def load(cls, file: Union[str, PurePath]) -> "BaseIndex":
        """Deserialize the index from disk"""
        raise NotImplementedError
    

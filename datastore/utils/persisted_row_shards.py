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

######################################################################################
# Sharding abstract classes. Shard by rows. Persists to some storage
# mechanism, which could include disk, remove file system (ffspec and
# s3filesystem or sqllalchemy database.

class PersistedRowShards(object):
  
  def __init__(self, shard_defs, fs=None, cache_dir=None):
    """
    :arg shard_defs: an array of dict, that has to have at a minimum the fileds, 'start_row', and 'end_row'
    :arg fs: the fsspec file system
    """
    shard_defs.sort(key=lambda a:a['start_row'])
    self.shard_defs = shard_defs
    self._shards = [None]*len(shard_defs) # TODO - use LRU cache so we can clear out db connecitons that are old or stale
    if fs is None:
      fs = os
    self.fs = fs
    if cache_dir is None: cache_dir = get_temporary_cache_files_directory()
    self.cache_dir = cache_dir

  def cache_shard_file(self, idx, f):
    shard_def = self.shard_defs[idx]
    fs = self.fs
    dataset_path = self.cache_dir
    if is_remote_filesystem(fs):
      f_dataset_path = extract_path_from_uri(f) 
      data_path = Path(dataset_path, f_dataset_path)
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
        with open(f.replace(".gz",".igz"), "wb") as f2:
          pickle.dump(fobj, f2, pickle.HIGHEST_PROTOCOL)
    return f

"""
    if False:
      cwd = os.getcwd()
      dir = os.path.abspath(os.path.dirname(f))
      f = os.path.basename(f)
      if dir:
        os.chdir(dir)
      with open(f.replace(".gz",".igz"), "rb") as f2:
        fobj = pickle.load(f2)
      os.chdir(cwd)
      return fobj
"""

  @lru_cache(100)
  def shards(self, idx):
    shard =  self._shard_by_idx(idx)
    shard.start_row = self.shard_defs['start_row']
    shard.end_row = self.shard_defs['end_row']
    return shard

  def _shard_by_idx(self, idx):
    raise NotImplementedError

  def delete_cache_files(self):
    pass


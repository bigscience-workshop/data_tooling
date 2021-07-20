#%%writefile data-tooling/datastore.py
#Copyright July 2021 Ontocord LLC. Licensed under Apache v2 https://www.apache.org/licenses/LICENSE-2.0
from dataclasses import asdict
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
from datasets.tasks import TaskTemplate
from datasets.table import InMemoryTable,  concat_tables
from datasets.dataset_dict import DatasetDict
from datasets import config
from datasets.filesystems import extract_path_from_uri, is_remote_filesystem
from datasets.utils import logging, map_nested
logger = logging.get_logger(__name__)
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



from getpass import getpass
import atexit, os, subprocess
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
import threading
import requests
import atexit
import uuid
import multiprocessing
from smart_open import open
import urllib
from datetime import datetime
from dask.distributed import Client
try:
  from datastore_utils import *
except:
  pass

import indexed_gzip as igzip
import os, multiprocessing, time, subprocess
import random
import socket
import glob
import copy
import itertools
from datetime import datetime, timedelta
import os
import signal
import atexit
import json
import warnings


#from flask import Flask;
import pandas as pd;
from pandas import DataFrame, read_csv
#from flask import Flask
import atexit
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from threading import Timer, Thread
from multiprocessing import Process
import subprocess
import requests
import os
import multiprocessing
from filelock import UnixFileLock, FileLock



### NOTE: dataset is a different package than datasets. We are using both packages.
### We want to have mutliple types of storage that ideally can be
### transported as a file transfer with an arrow dataset (perhaps a tar file?). So if we
### have <signature>.arrow, we may have fts_<signature>.db (for full
### text indexing sqlite database) and <signature>.db (sqlite database), and
### <siganture>.mmap (mmap file reprsenting a tensor), and
### <singature>.igz (if we wish to store some portion of the text
### columns in igzip format for compression and legacy purposes.


### A note about naming: datasets uses the terms features and columns interchangably.


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    
# FileLock does not really work for gdrive. So we create a SharedFileLock instead, which is not guranteed to lock a file.
# This is because if two nodes are trying to write to the same file, gdrive will create a file(1).txt as the other file being written too.
# Not thread or multiprocessing safe either.
class SharedFileLock(UnixFileLock):
    def __init__(self, lock_file, timeout = -1, locks=None):
        super().__init__(lock_file, timeout)
        self.locks=locks
        self.locked=False

    def __enter__(self):
        if self.locks is not None and self._lock_file not in self.locks:
            self.locks[self._lock_file] = 1
            self.acquire()
            self.locked=True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.locked: 
          self.release()
          if self.locks is not None and self._lock_file in self.locks:
              del self.locks[self._lock_file]
          self.locked=False
        return None

def preexec(): # Don't forward signals.
        os.setpgrp()

#code to do distributed context over dask with an optional streamlit app attached. made to work through ngrok. 
# ngrok code is based on https://github.com/gstaff/flask-ngrok which is licensed under Apache v2

#TODO: make to work withough ngrok.
#TODO: streamlit does tracking. For privacy and security concern, we need to turn this off.
# this class provides Dask and Streamlit through ngrok.
# for example:
#
# DistributedContext.start(shared_scheduler_file="<my scheduler file>", token=getpass("token:"), clear_streamlit_app=True)
# st = DistributedContext()
#
# if you do not provide a custom <my app file>.py file, an app file will automatically be created, and the file name stored in DistributedContext.streamlit_app_file

class DistributedContext:
  ngrok = None
  dask_node_id = None
  dask_nodes = {}
  dask_client = None
  dask_scheduler_file = None
  streamlit_app_file = None
  streamlit_process = None

  def __init__(self, st=None):

    self._print = print
    self._isnotebook = DistributedContext.isnotebook()
    if st is not None:
      self.st = st
    elif self._isnotebook:
      try:
        import streamlit as st
        if st._is_running_with_streamlit:
          self.st = st
        else:
          self.st = None
      except:
        raise RuntimeError("streamlit not found. Install with pip install streamlit")
    else:
      self.st = None


  @staticmethod
  def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell is not None and "Interactive" not in shell:
          return True
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
  @staticmethod
  def log_to_app(args):
    with FileLock(DistributedContext.streamlit_app_file+".lock"):
      with open(DistributedContext.streamlit_app_file, "a+") as f:
        if isinstance(item, tuple):
          for an_item in item: 
                f.write(f"st.write(\"\"\""+str(an_item)+"\"\"\")\n")
        else:
              f.write(f"st.write(\"\"\""+str(item)+"\"\"\")\n")
        f.flush()

  def print(self, *args, **kw):
    if DistributedContext.dask_client: 
      DistributedContext.dask_client.submit(DistributedContext.log_to_app, args, worker="worker_0-0", pure=False).result()
    self._print(*args, **kw)

  def print_streamlit(self, *args, **kw):
    self.st.write(*args)

  #monkey patch print
  def __enter__(self):
    global print
    if self.st is None:
      if print not in (self.print_streamlit, self.print):  
        if print != self._print:
          self._print = print
        print = self.print
    else:
      if print not in (self.print_streamlit, self.print):  
        if print != self._print:
          self._print = print
        print = self.print_streamlit

  def __exit__(self, exc_type, exc_val, exc_tb):
    global print
    print = self._print


  @staticmethod
  def reload_dask_nodes(shared_scheduler_file, time_out=43200):
    with SharedFileLock(shared_scheduler_file+".lock"):
      file = shared_scheduler_file.split(".")
      files = glob.glob(".".join(file[:len(file)-1])+"*."+file[-1])
      has_timeout = False
      dask_nodes={}
      for a_file in wait_until_files_loaded(files):
        with open(a_file, "r") as f:
          for line in f.read().split("\n"):
            if not line.strip():
              continue
            node_id, address, last_heartbeat = line.split("\t")
            node_id = int(node_id)
            last_heartbeat = datetime.fromtimestamp(float(last_heartbeat))
            if datetime.timestamp(datetime.now()) - datetime.timestamp(last_heartbeat) > time_out: 
              print (f"node {node_id} has timed out. removing from scheduler")
              has_timeout = True
            else:
              dask_nodes[node_id] = (address, last_heartbeat)
      DistributedContext.dask_nodes = dask_nodes
      if has_timeout:
        with open(shared_scheduler_file, "w") as f: 
          for key, val in DistributedContext.dask_nodes.items():
            f.write(str(key)+"\t"+str(val[0])+"\t"+str(val[1])+"\n")
      if DistributedContext.dask_node_id == 0 and len(files) > 1:
        # let's do a sanity check to see if there are more than one shared_scheduler_file
        # this might happen because locking is perfect over shared drives. 
        # if there are duplicates, we will merge the files

        for a_file in wait_until_files_loaded(files):
          if os.path.exists(a_file):
            os.unlink(a_file) 
        with open(shared_scheduler_file, "w") as f: 
          for key, val in DistributedContext.dask_nodes.items():
            f.write(str(key)+"\t"+str(val[0])+"\t"+str(val[1])+"\n")

  @staticmethod
  def start(shared_scheduler_file=None, token=None, streamlit_port=8501,  streamlit_app_file=None, hostname=None, import_code="", header_html="", num_procs=4, clear_streamlit_app=False):
    DistributedContext.stop()
    if token:
      DistributedContext.write_authtoken(token)
    if shared_scheduler_file is None:
      shared_scheduler_file = DistributedContext.dask_scheduler_file
    if shared_scheduler_file is None:
      raise RuntimeError("must provide a shared_scheduler_file to start dask")
    generic_streamlit_app = False
    if streamlit_app_file is None:
      generic_streamlit_app = True
      streamlit_app_file= str(Path(tempfile.gettempdir(), "app.py"))
    DistributedContext.streamlit_app_file = streamlit_app_file
    if shared_scheduler_file is None:
      shared_scheduler_file = DistributedContext.dask_scheduler_file
    if DistributedContext.dask_scheduler_file is not None and shared_scheduler_file != DistributedContext.dask_scheduler_file:
      raise RuntimeError(f"dask_scheduler_file already set. expected {DistributedContext.dask_scheduler_file}")
    DistributedContext.dask_scheduler_file = shared_scheduler_file
    if shared_scheduler_file is None:
        raise RuntimeError("no shared_scheduler_file provided")
    DistributedContext.reload_dask_nodes(shared_scheduler_file)
    if 0 in DistributedContext.dask_nodes:
        print("warning: dask scheduler already started. restarting.")
    DistributedContext.dask_node_id = 0
    DistributedContext.dask_client = Client(n_workers=num_procs, name="worker_0")
    dask_port = int(str(DistributedContext.dask_client).split("tcp://")[1].strip().split()[0].strip("' ").split(":")[-1].strip())
    addresses = DistributedContext._run_ngrok(streamlit_port, tcp_port=dask_port, hostname=hostname, region="us")
    DistributedContext.dask_nodes[DistributedContext.dask_node_id] = ([a for a in addresses if a.startswith("tcp:")][0], datetime.timestamp(datetime.now()))
    print ("dask schduler running on", DistributedContext.dask_nodes[DistributedContext.dask_node_id][0])
    with SharedFileLock(shared_scheduler_file+".lock"):
        with open(shared_scheduler_file, "w") as f: 
          for key, val in DistributedContext.dask_nodes.items():
            f.write(str(key)+"\t"+str(val[0])+"\t"+str(val[1])+"\n")
    webaddr = [a for a in addresses if a.startswith("https:")][0]
    if generic_streamlit_app and (not os.path.exists(streamlit_app_file) or clear_streamlit_app):
      DistributedContext.create_streamlit_app(streamlit_app_file, import_code=import_code, header_html=header_html)
    DistributedContext.streamlit_process = subprocess.Popen(('streamlit', 'run', streamlit_app_file), preexec_fn = preexec)
    atexit.register(DistributedContext.streamlit_process.terminate)
    print (f"streamlit server running on {webaddr}")
    #time.sleep(5)
    DistributedContext.reload_dask_nodes(shared_scheduler_file)
    return DistributedContext.dask_client 

  @staticmethod
  def launch_dask_node(shared_scheduler_file, num_procs=4, time_out=43200): # timeout in 12 hours
    if shared_scheduler_file is None:
      shared_scheduler_file = DistributedContext.dask_scheduler_file
    if DistributedContext.dask_scheduler_file is not None and shared_scheduler_file != DistributedContext.dask_scheduler_file:
      raise RuntimeError(f"dask_scheduler_file already set. expected None or {DistributedContext.dask_scheduler_file}")
    DistributedContext.dask_scheduler_file = shared_scheduler_file
    if DistributedContext.dask_node_id is not None and DistributedContext.dask_client is not None:
      return DistributedContext.dask_client
    DistributedContext.reload_dask_nodes(shared_scheduler_file, time_out=time_out)
    if 0 not in DistributedContext.dask_nodes:
        raise RuntimeError("no scheduler started. try: st = DistributedContext().start(start_dask=True, shared_scheduler_file=<shared scheduler file>.tsv) ")
    if DistributedContext.dask_node_id == 0:
          DistributedContext.dask_client = Client(DistributedContext.dask_nodes[0][0])
          return DistributedContext.dask_client, DistributedContext.dask_node_id
    else:
          address = DistributedContext.dask_nodes[0][0]
          DistributedContext.dask_node_id = len(DistributedContext.dask_nodes)
          # todo, pipe stderr and check if there are errors.
          dask = subprocess.Popen(["dask-worker", address, '--name', f"worker_{DistributedContext.dask_node_id}", "--nprocs", num_proces, "--nthreads", "1", "--no-dashboard"], preexec_fn = preexec) 
          atexit.register(dask.terminate)
          #!dask-worker $address --name $dask_node_id --nprocs $num_procs --nthreads 1  --no-dashboard 
          # if there is an error in connecting to the scheduler, then the scheduler has probably died and we need to create a new scheduler with this node.
          DistributedContext.dask_nodes[DistributedContext.dask_node_id] = (DistributedContext.dask_node_id, datetime.timestamp(datetime.now()))
          with SharedFileLock(shared_scheduler_file+".lock"):
            with open(shared_scheduler_file, "w") as f: 
              for key, val in DistributedContext.dask_nodes.items():
                f.write(str(key)+"\t"+str(val[0])+"\t"+str(val[1])+"\n")
          DistributedContext.reload_dask_nodes(shared_scheduler_file, time_out=time_out)
          DistributedContext.dask_client = Client(DistributedContext.dask_nodes[0][0])
          return DistributedContext.dask_client, DistributedContext.dask_node_id

  @staticmethod
  def stop():
    if DistributedContext.ngrok is not None:
      DistributedContext.ngrok.terminate()
    DistributedContext.ngrok = None
    if DistributedContext.streamlit_process is not None:
      DistributedContext.streamlit_process.terminate()
    DistributedContext.streamlit_process = None
    if DistributedContext.dask_client is not None:  
      DistributedContext.dask_client.shutdown()
    DistributedContext.dask_client = None
    DistributedContext.dask_node_id = None
    DistributedContext.dask_nodes = {}
    DistributedContext.dask_client = None
    DistributedContext.dask_scheduler_file = None
    
  @staticmethod
  def create_streamlit_app(file, import_code="", header_html=""):
      with FileLock(file+".lock"):
        with open(file, "w+") as f:
          f.write("""
import os
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
"""+import_code+"""
hide_menu_style = \"\"\"
              <style>
              #MainMenu {visibility: hidden;}
              </style>
\"\"\"
def create_header():
      global hide_menu_style
      st.write (hide_menu_style)
create_header()
### begin app code 
  """)
          f.flush()


  @staticmethod
  def write_authtoken(token):
    if not os.path.exists("/root/.ngrok2/"):
      os.mkdir("/root/.ngrok2")
    with open("/root/.ngrok2/ngrok.yml", "w") as f:
      f.write("authtoken: "+token+"\n")

  #todo - tls
  @staticmethod
  def _run_ngrok(port, tcp_port=None, hostname=None, region="us"):
    command = DistributedContext._get_command()
    ngrok_path = str(Path(tempfile.gettempdir(), "ngrok"))
    DistributedContext._download_ngrok(ngrok_path)
    executable = str(Path(ngrok_path, command))
    os.chmod(executable, 0o777)
    if not os.path.exists("/root/.ngrok2/"):
      os.mkdir("/root/.ngrok2")
    with open("/root/.ngrok2/app.yml", "w") as f:
      f.write(f"region: {region}\n")
      f.write(f"tunnels:\n")
      f.write(f" website:\n")
      f.write(f"  addr: {port}\n")
      if hostname is not None:
        f.write(f"  hostname: {hostname}\n")
      f.write(f"  proto: http\n")
      if tcp_port is not None:
        f.write(f" tcp_port:\n")
        f.write(f"  addr: {tcp_port}\n")
        f.write(f"  proto: tcp\n")
      
    if not os.path.exists("/root/.ngrok2/ngrok.yml"):
      ngrok = DistributedContext.ngrok = subprocess.Popen([executable, 'start', '-config', '/root/.ngrok2/app.yml', 'tcp_port', 'website'], preexec_fn = preexec)
    else:
      ngrok = DistributedContext.ngrok = subprocess.Popen([executable, 'start', '-config', '/root/.ngrok2/ngrok.yml', '-config', '/root/.ngrok2/app.yml', 'tcp_port', 'website'], preexec_fn = preexec)
    atexit.register(ngrok.terminate)
    localhost_url = "http://localhost:4040/api/tunnels"  # Url with tunnel details
    tunnel_urls = []
    for i in range(10):
      time.sleep(5)
      try:
        tunnel_url = requests.get(localhost_url).text  # Get the tunnel information
        j = json.loads(tunnel_url)
        for a_tunnel in  j['tunnels']:
          tunnel_url = a_tunnel['public_url'] #.replace("http", "https")
          tunnel_urls.append(tunnel_url)
        if tunnel_urls:
          break
      except:
        pass
    return tunnel_urls


  @staticmethod
  def _get_command():
      system = platform.system()
      if system == "Darwin":
          command = "ngrok"
      elif system == "Windows":
          command = "ngrok.exe"
      elif system == "Linux":
          command = "ngrok"
      else:
          raise Exception("{system} is not supported".format(system=system))
      return command

  @staticmethod
  def _download_ngrok(ngrok_path):
      if Path(ngrok_path).exists():
          return
      system = platform.system()
      if system == "Darwin":
          url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-darwin-amd64.zip"
      elif system == "Windows":
          url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-windows-amd64.zip"
      elif system == "Linux":
          url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip"
      else:
          raise Exception(f"{system} is not supported")
      download_path = DistributedContext._download_file(url)
      with zipfile.ZipFile(download_path, "r") as zip_ref:
          zip_ref.extractall(ngrok_path)

  @staticmethod
  def _download_file(url):
      local_filename = url.split('/')[-1]
      r = requests.get(url, stream=True)
      download_path = str(Path(tempfile.gettempdir(), local_filename))
      with open(download_path, 'wb') as f:
          shutil.copyfileobj(r.raw, f)
      return download_path

# some utils

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
  return

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


"""
class MetadataMixin:

    @classmethod
    def rowid_key_name():
        raise NotImplementedError()

    @classmethod
    def extract_rowid(batch):
        raise NotImplementedError()

    @classmethod
    def section_key_name():
        raise NotImplementedError()

    @classmethod
    def extract_section_labels(batch):
        raise NotImplementedError()

    @classmethod
    def url_key_name():
        raise NotImplementedError()
        
    @classmethod
    def extract_url(batch):
        raise NotImplementedError()

    @classmethod
    def outlinks_key_name():
        raise NotImplementedError()
                
    @classmethod
    def extract_outlinks(batch):
        raise NotImplementedError()

    @classmethod
    def publication_timestamp_key_name():
        raise NotImplementedError()
                
    @classmethod
    def extract_publication_timestamp(batch):
        raise NotImplementedError()

    @classmethod
    def collection_timestamp_key_name():
        raise NotImplementedError()
                
    @classmethod
    def extract_collection_timestamp(batch):
        raise NotImplementedError()

    @classmethod
    def uuid_hash_key_name():
        raise NotImplementedError()
                
    @classmethod
    def create_uuid_hash(batch):
        raise NotImplementedError()

    @classmethod
    def main_langid_key_name(batch):
        raise NotImplementedError()

    @classmethod
    def extract_main_langid(batch):
        raise NotImplementedError()

    @classmethod
    def other_langid_key_name(batch):
        raise NotImplementedError()

    @classmethod
    def extract_other_langid(batch):
        raise NotImplementedError()

    @classmethod
    def pii_multiple_key_names():
      raise NotImplementedError()

    @classmethod
    def detect_and_process_pii(batch):
        raise NotImplementedError()


    # a meta-data field to note the dataset this came from and the split

    # a meta-data field to note that the data is synehtic and how?

    # a meta-data field to note the history transformations to get to this row?

#this class will automatically create a logs and metrics table.
#it should be a attached to datastore in its pipeline for processing. 
class AuditLogAndMetricsMixin:

    def __init__(self, connection_url):
      self.connection_url = connection_url
      db = DatabaseExt(connection_url)
      db['__logs']
      db['__metrics']
      db['__carbon_usage']
      # we don't store anything in the object that can't be persisted.
      # all data is stored in the backend database itself.
  
    @classmethod
    def log_step(batch):
        raise NotImplementedError()

# this class will automatically create an ontology table. it is used to among other things do NER/PII meta-data tagging.
# this class has methods for accessing and manipulating an hypernym/hyponym or is-a/is-isntance of ontology
# includes methods such as lcs, and language, and examples reference (dataset, split, rowid, uuid)
class OntologyMixin:

    def __init__(self, connection_url):
      self.connection_url = connection_url
      db = DatabaseExt(connection_url)
      db['__ontology']
      # we don't store anything in the object that can't be persisted.

"""

def np_mmap(path, dtype, shape):
  if os.path.exists(path):
    return np.memmap(filename=path, mode="r+", dtype=np.dtype(dtype), shape=tuple(shape))
  else:
    return np.memmap(filename=path, mode="w+", dtype=np.dtype(dtype), shape=tuple(shape))
      
class FeaturesWithViews(Features):
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

class Datastore(Dataset): 
    """
    A class that wraps a Huggingface arrow based Dataset to provide some optimized reading and *writing* in various persistance backends. 
    Currently provides support for features bound to memmap, indexed gzip (igzip) file, and sqlalchemy databases.
    """
        
    def __repr__(self):
        ret = FeaturesWithViews(self._info.features)
        ret.views_map = {} if not hasattr(self, "views_map") else self.views_map
        return f"Datastore({{\n    features: {ret},\n    num_rows: {self.num_rows}\n}})"
    

    @classmethod
    def from_dataset(cls, dataset, template_datastore=None, views_map=None, id_feature=None, metadata_manager=None, id_idx_identical=None,):
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
        
        if  hasattr(dataset, "id_idx_identical"):
          self.id_idx_identical = dataset.id_idx_identical
        elif  id_idx_identical is not None:
          self.id_idx_identical = id_idx_identical
        elif hasattr(template_datastore, "id_idx_identical"):
          self.id_idx_identical = template_datastore.id_idx_identical
        else:
          self.id_idx_identical = True

        if  hasattr(dataset, "metadata_manager"):
          self.metadata_manager = dataset.metadata_manager
        elif  metadata_manager is not None:
          self.id_idx_identical = metadata_manager
        elif hasattr(metadata_manager, "metadata_manager"):
          self.metadata_manager = template_datastore.metadata_manager
        else:
          self.metadata_manager = None

        if  hasattr(dataset, "id_feature"):
          self.id_feature = dataset.id_feature
        elif  id_feature is not None:
          self.id_feature = id_feature
        elif hasattr(metadata_manager, "id_feature"):
          self.id_feature = template_datastore.id_feature
        else:
          self.id_feature = "id"

        if  hasattr(dataset, "views_map"):
          self.views_map = copy.deepcopy(dataset.views_map)
        elif  views_map is not None:
          self.views_map = copy.deepcopy(views_map)
        elif hasattr(metadata_manager, "views_map"):
          self.views_map = copy.deepcopy(template_datastore.views_map)
        else:
          self.views_map = {}

        return self


    def apply_metadata_manager(self, metadata_manager=None):
      if hasattr(self, 'metadata_manager') and self.metadata_manager not in (None, metadata_manager):
          raise RuntimeError(f"attempting to reset the metadta_manager to {metadata_manager}")
      elif metadata_manager is not None:
          self.metadata_manager = metadata_manager
      if self.metadata_manager is not None:
        self = self.map(self.metadata_manager,  batch_size=batch_size, batched=True, num_proc=num_proc)
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
    def _move_to_mmap_col(batch, src_feature, id_feature, path, dtype, shape):
        ret = np_mmap(path, dtype, shape)
        ret[batch[id_feature]] = batch[dst_feature_view]

    @classmethod
    def from_mmap(cls,  feature_view, shape, path=None, dtype='float32', dtype_str_len=100, id_feature="id", batch_size=1000, num_proc=4, metadata_manager=None):
      return cls.from_dict({}).add_mmap(feature_view=feature_view, shape=shape, path=path, dtype=dtype, dtype_str_len=dtype_str_len, id_feature=id_feature, batch_size=batch_size, num_proc=num_proc, metadata_manager=metadata_manager)

    def move_to_mmap(self, src_feature, dst_feature_view=None, shape=None, path=None, dtype='float32', dtype_str_len=100, id_feature="id", batch_size=1000, num_proc=4, metadata_manager=None):
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
      self.add_mmap(feature_view=dst_feature_view, shape=shape, path=path, dtype=dtype, id_feature=id_feature, batch_size=batch_size, num_proc=num_proc)
      val = self.views_map[dst_feature_view]
      self.map(Datastore._move_to_mmap_col, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'src_feature':src_feature, 'id_feature':id_feature, 'path': val['path'], 'dtype': val['dtype'], 'shape':shape})
      self= self.remove_columns(src_feature)
      if hasattr(self, 'metadata_manager') and self.metadata_manager not in (None, metadata_manager):
          raise RuntimeError(f"attempting to reset the metadta_manager to {metadata_manager}")
      elif metadata_manager is not None:
          self.metadata_manager = metadata_manager
      if metadata_manager is not None:
        self = self.map(metadata_manager,  batch_size=batch_size, batched=True, num_proc=num_proc)
      return self

    #mapping a feature/columun to a memmap array accessed by row
    def add_mmap(self, feature_view, shape, path=None, dtype='float32', dtype_str_len=100, id_feature="id", batch_size=1000, num_proc=4, metadata_manager=None):
      if not hasattr(self, 'views_map'): self.views_map = {}
      if hasattr(self, 'id_feature') and self.id_feature != id_feature:
        raise RuntimeError(f"attempting to reset the index to {id_feature}")
      else:
        self.id_feature = id_feature
      if hasattr(self, 'metadata_manager') and self.metadata_manager not in (None, metadata_manager):
          raise RuntimeError(f"attempting to reset the metadta_manager to {metadata_manager}")
      elif metadata_manager is not None:
          self.metadata_manager = metadata_manager
      if not self.cache_files:
        dataset_path = get_temporary_cache_files_directory()
      else:  
        dataset_path = os.path.dirname(self.cache_files[0]['filename'])
      if path is None:
          path = os.path.abspath(os.path.join(dataset_path, feature_view+".mmap"))
      shape = list(shape)
      shape[0] = max(shape[0], len(self))
      if id_feature not in self.features:
        if len(self) == 0 and shape[0] > 0:
            self = Datastore.from_dataset(Dataset.from_dict({id_feature: range(shape[0])}), self)
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
      self.views_map[feature_view] = {'type':"mmap", 'path': path,  'dtype': dtype, 'shape': shape}
      if metadata_manager is not None:
        self = self.map(metadata_manager,  batch_size=batch_size, batched=True, num_proc=num_proc)
      return self


    @classmethod
    def from_igzip(cls, feature_view, path,  id_feature="id", batch_size=1000, num_proc=4, metadata_manager=None):
      return cls.from_dict({}).add_igzip(feature_view=feature_view, path=path,  id_feature=id_feature, batch_size=batch_size, num_proc=num_proc, metadata_manager=metadata_manager)

    #mapping a feature/columun to an indexed gzip file accessed by line 
    def add_igzip(self, feature_view, path,  id_feature="id", batch_size=1000, num_proc=4, metadata_manager=None):
      if not hasattr(self, 'views_map'): self.views_map = {}
      if hasattr(self, 'id_feature') and self.id_feature != id_feature:
        raise RuntimeError(f"attempting to reset the index to {id_feature}")
      else:
        self.id_feature = id_feature
      if hasattr(self, 'metadata_manager') and self.metadata_manager not in (None, metadata_manager):
          raise RuntimeError(f"attempting to reset the metadta_manager to {metadata_manager}")
      elif metadata_manager is not None:
          self.metadata_manager = metadata_manager
      fobj = self._get_igzip_fobj(path)
      if id_feature not in self.features:
          if len(self) == 0:
            self = Datastore.from_dataset(Dataset.from_dict({id_feature: range(len(fobj))}), self)
            ids = dict([(a,1) for a in range(len(self))])
            self.id_idx_identical = True
          else:
            print ("adding idx")
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
      self.views_map[feature_view] = {'type':"igzip", 'path': path}
      if metadata_manager is not None:
        self = self.map(metadata_manager,  batch_size=batch_size, batched=True, num_proc=num_proc)
      return self


    def move_to_sql(self, src_feature_to_dst_feature_map, table_name=None, connection_url=None,  id_feature="id",  batch_size=1000, num_proc=4, metadata_manager=None):
      if table_name is None:
          #print (self.info.builder_name, self.info.config_name)
          table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}"
      if not connection_url:
          connection_url="sqlite:///"+self.cache_files[0]['filename'].replace(".arrow", ".db")
      table = Datastore._get_db_table(self, table_name, connection_url)
      if type(src_feature_to_dst_feature_map) is list:
        src_feature_to_dst_feature_map = dict(src_feature_to_dst_feature_map)
      elif type(src_feature_to_dst_feature_map) is str:
        src_feature_to_dst_feature_map = {src_feature_to_dst_feature_map: src_feature_to_dst_feature_map}
      feature_view = []
      for src_feature, dst_feature_view in list(src_feature_to_dst_feature_map.items()):
        if src_feature == dst_feature_view:
          self = self.rename_column(src_feature, "__tmp__"+src_feature)
          src_feature_to_dst_feature_map["__tmp__"+src_feature] = dst_feature_view
          del src_feature_to_dst_feature_map[src_feature]
          src_feature = "__tmp__"+src_feature
        value = self[0][src_feature]
        if type(value) is str: #we don't want to save as json type just in case
            value="**"
        dtype = table.db.types.guess(value)
        feature_view.append((dst_feature_view, dtype))
      self.add_sql(feature_view=feature_view, table_name=table_name, connection_url=connection_url, id_feature=id_feature, batch_size=batch_size, num_proc=num_proc)
      self = self.map(Datastore.upsert_sql_from_batch, batch_size=batch_size, batched=True, num_proc=1 if connection_url=="sqlite://" else num_proc, fn_kwargs={'views_map':self.views_map, 'id_feature':id_feature, 'src_feature_to_dst_feature_map': src_feature_to_dst_feature_map})
      self = self.remove_columns(src_feature)
      if hasattr(self, 'metadata_manager') and self.metadata_manager not in (None, metadata_manager):
          raise RuntimeError(f"attempting to reset the metadta_manager to {metadata_manager}")
      elif metadata_manager is not None:
          self.metadata_manager = metadata_manager
      if metadata_manager is not None:
        self = self.map(metadata_manager,  batch_size=batch_size, batched=True, num_proc=num_proc)
      return self

    @classmethod
    def from_sql(cls,  feature_view, table_name, connection_url, dtype="str", id_feature="id",  batch_size=1000, num_proc=4, metadata_manager=None):
      return cls.from_dict({}).add_sql(feature_view=feature_view, table_name=table_name, connection_url=connection_url, dtype=dtype, id_feature=id_feature, batch_size=batch_size, num_proc=num_proc, metadata_manager=metadata_manager)

    # mapping one or more columns/features to a sql database. creates a sqlalchmey/dataset dynamically with id_feature as the primary key. 
    # TODO: remember to strip passwords from any connection_url. passwords should be passed as vargs and added to the conneciton url dynamically
    # passwords should not be perisisted.
    # NOTE: this dataset will not automatically change if the database changes. call this method again to sync the size and schema
    def add_sql(self, feature_view=None, table_name=None, connection_url=None, dtype="str", id_feature="id",  batch_size=1000, num_proc=4, metadata_manager=None):
        if not hasattr(self, 'views_map'): self.views_map = {}
        if hasattr(self, 'id_feature') and self.id_feature != id_feature:
          raise RuntimeError(f"attempting to reset the index to {id_feature}")
        else:
          self.id_feature = id_feature
        if hasattr(self, 'metadata_manager') and self.metadata_manager not in (None, metadata_manager):
          raise RuntimeError(f"attempting to reset the metadta_manager to {metadata_manager}")
        elif metadata_manager is not None:
          self.metadata_manager = metadata_manager
        if table_name is None:
          #print (self.info.builder_name, self.info.config_name)
          table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}"
          #print (table_name)
        if not connection_url:
          connection_url="sqlite:///"+self.cache_files[0]['filename'].replace(".arrow", ".db")
        if type(feature_view) is str:
          feature_view = [(feature_view, dtype)]
        if type(feature_view) is dict:
          feature_view =  list(feature_view.items())
        table = self._get_db_table(table_name, connection_url)
        if not feature_view and table.columns:
            feature_view = table.columns
        elif not feature_view:
            raise RuntimeError(f"No feature_view(s) and no column definition for table view {table_name}")
        table_ids = table.find(_columns=id_feature)
        if id_feature not in self.features:
          if len(self) == 0 and table_ids:
            self = Datastore.from_dataset(Dataset.from_dict({id_feature: range(max([id[id_feature] for id in table_ids]))}), self)
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
            self.views_map[col] = {'type':'sql', 'connection_url': connection_url, 'table_name': table_name}
        if metadata_manager is not None:
          self = self.map(metadata_manager,  batch_size=batch_size, batched=True, num_proc=num_proc)
        return self
    

    #filter_sql uses a sql query on columns that are mapped to sql. 
    #could be faster than doing a normal "filter"
    #the parameters are the same as the "find" method from dataset.Table.
    #example: dataset.filter_sql(lang='ru') will return those items in the dataset that has the language 'ru'.
    #NOTE: fts not yet working.
    def filter_sql(self, **kwargs):
      if not hasattr(self, 'views_map'): self.views_map = {}
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
        for feature_view, val in self.views_map.items():
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
    # instead, a[0] might return {'id': 10, 'mmap_embed': <array correponding to the 10th location in the mmap file>}. 
    # To get dataset items by 'id', use either filter or filter_sql.
    # check the property id_idx_identical to determine if the id corresponds to the index of the table.
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
            key = self.id_feature
        missing=[]
        if format_columns:
            for c in copy.copy(format_columns):
                if c in self.views_map:
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
    def upsert_sql_from_batch(batch, views_map, id_feature, src_feature_to_dst_feature_map):
      sql_results={}
      for src_feature, dst_feature in src_feature_to_dst_feature_map.items() if src_feature_to_dst_feature_map is not None else zip(batch.keys(),batch.keys()):
        if views_map.get(dst_feature):
          val = views_map[dst_feature]
          if val['type'] == 'sql':
            sql_results[(val['table_name'], val['connection_url'])] = sql_results.get((val['table_name'], val['connection_url']),[])+[(src_feature, dst_feature)]
      for key, features in sql_results.items():
        table_name, connection_url = key
        db = DatabaseExt(connection_url)
        with db:
            table = db[table_name]
            batch2 = []
            for i in range(len(batch[id_feature])):
              batch2.append(dict([(feature[1], batch[feature[0]][i]) for feature in features+[(id_feature,id_feature)]]))               
            try:
              table.insert_many(batch2)
            except:
              batch2 = []
              for i in range(len(batch[id_feature])):
                batch2.append(dict([(feature[1], batch[feature[0]][i]) for feature in features+[(id_feature,id_feature)]]))    
              table.update_many(batch2, [id_feature])
            batch2 = None

    PERSIST_IN_ARROW = 0
    STATIC_VIEWS = 1
    UPDATE_VIEWS = 2

    @staticmethod
    def map_fn_with_indices_and_handle_views(batch, indices, map_fn, fn_kwargs, handle_views, views_map, id_feature):
      ret = map_fn(batch, indices, **fn_kwargs)
      if ret is not None and views_map:
        if views_map and id_feature not in ret:
          raise RuntimeError(f"Datstore returned from map must have an {id_feature} column to link to views.")
        if handle_views != DataStore.PERSIST_IN_ARROW:
          for key in views_map:
            if handle_views == Datastore.UPDATE_VIEWS:
              if val['type'] == 'mmap':
                  mmap_array = np_mmap(val['path'], val['dtype'], val['shape'])
                  mmap_array[batch[id_feature]] = batch[feature]                     
              elif val['type'] == 'igzip':
                  raise RuntimeError("cannot update an igzip file")
            elif handle_views == Datastore.STATIC_VIEWS:
              if key in ret:
                del ret[key]
          if handle_views == 2: Datastore.upsert_sql_from_batch(ret, views_map, id_feature, None)
      return ret

    @staticmethod
    def map_fn_and_handle_views(batch, map_fn, fn_kwargs, handle_views, views_map, id_feature):
      ret = map_fn(batch, **fn_kwargs)
      if ret is not None and views_map:
        if views_map and id_feature not in ret:
          raise RuntimeError(f"Datstore returned from map must have an {id_feature} column to link to views.")
        if handle_views != Datastore.PERSIST_IN_ARROW:
          for key in views_map:
            if handle_views == Datastore.UPDATE_VIEWS:
              if val['type'] == 'mmap':
                  mmap_array = np_mmap(val['path'], val['dtype'], val['shape'])
                  mmap_array[batch[id_feature]] = batch[feature]                     
              elif val['type'] == 'igzip':
                  raise RuntimeError("cannot update an igzip file")
            elif handle_views == Datatsore.STATIC_VIEWS:
              if key in ret:
                del ret[key]
          if handle_views == 2: Datastore.upsert_sql_from_batch(ret, views_map, id_feature, None)
      return ret

    #parameter handle_views tells us how to handle views. 
    #PERSIST_IN_ARROW - all data returned will be persisted to arrow storage and not views. this will detach all views.
    #STATIC_VIEWS - keep the views attached to external storage without change. *default*
    #UPDATE_VIEWS - update views based on what is returned by the map function. this will create a side-effect.
    #WARNING might create an unepxected side-effect on caching.  Use caching with cuation when editing views.
    #parameter update_metadata tells us to run the data through the metdata-manager again to update the metdata. 
    #if you set both handle_views=Datastore.UPDATE_VIEWS and update_metdata=True, the metadata will updated by the metadata manager only, 
    #and any updates to the metadata returned by your map funciton will be ignored.
    
    #consider whether we just want to attach a callback for label functions, similar to snorkel LF functions.

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
        handle_views=STATIC_VIEWS
    ) -> "Datastore":
      if not hasattr(self, 'views_map'): self.views_map = {}
      views_map= copy.deepcopy(self.views_map)
      for column in remove_columns if remove_columns is not None else []:
          if column in views_map:
              del views_map[column]
      if handle_views != Datastore.PERSIST_IN_ARROW:
        fn_kwargs = {'fn_kwargs': fn_kwargs, 'views_map': views_map, 'map_fn': function, 'handle_views': handle_views, 'id_feature': self.id_feature}
        if with_indices:
            function = Datastore.map_fn_with_indices_and_handle_views
        else:
            function = Datastore.map_fn_and_handle_views
      ret = super().map(function=function, with_indices=with_indices, input_columns=input_columns,
                     batched=batched, batch_size=batch_size, drop_last_batch=drop_last_batch, 
                     remove_columns=remove_columns, keep_in_memory=keep_in_memory, 
                     load_from_cache_file=load_from_cache_file, cache_file_name=cache_file_name,
                     writer_batch_size=writer_batch_size, features=features,
                     disable_nullable=disable_nullable, fn_kwargs=fn_kwargs,
                     num_proc=num_proc, suffix_template=suffix_template,
                     new_fingerprint=new_fingerprint)
      for column in remove_columns if remove_columns is not None else []:
         if column in self.views_map and column in ret:
           print (f"warning: the map function returned a column {column} which is the same as a detached view. this column will be persisted to arrow.")
      return Datastore.from_dataset(ret, self, views_map=views_map)

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
        if not hasattr(self, 'views_map'): self.views_map = {}
        views_map= copy.deepcopy(self.views_map)
        for column in remove_columns if remove_columns is not None else []:
            if column in views_map:
                del views_map[column]
        return Datastore.from_dataset(ret, self, views_map=views_map)


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
    def add_batch(self, batch: dict, new_fingerprint: str):
        """Add batch to Dataset.

        Args:
            batch (dict): batch data to be added.

        Returns:
            :class:`Dataset`
        """
        if not hasattr(self, 'views_map'): self.views_map = {}
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


    def save_to_disk(self, dataset_path: str, fs=None, move_files=False):
      # move_files means delete the old files as we create the new files to dataset_path.
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
            if 'connection_url' in value:
              if "sqlite:///" in value['connection_url']:
                src = value['connection_url'].replace("sqlite:///", "")
                db = Datastore.db_connection[value['connection_url']]
                db.close()
                del  Datastore.db_connection[value['connection_url']]
                db = None
                for key in list(Datastore.db_table.keys()):
                  if key[1] == value['connection_url']:
                    del Datastore.db_table[key]
                value['connection_url'] = "sqlite:///"+Path(src).name
              else:
                continue
            else:
              src = value['path']
              value['path'] = Path(src).name
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
                "id_idx_identical",
                "id_feature",
                "metadata_manager"
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
          #NEED TO TAKE CARE OF CASE WHERE WE SAVED TO TEMP TABLE
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
        ret.id_idx_identical =  state.get("id_idx_identical")
        fs = fsspec.filesystem("file") if fs is None else fs
        for key, value in list(ret.views_map.items()):
            if 'connection_url' in value:
              if "sqlite:///" in value['connection_url']:
                src = Path(value['connection_url'].replace("sqlite:///", "")).name
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
                if value['type'] == 'igzip':
                  src_dataset_path2 = src_dataset_path2.replace(".gz", ".igz")
                  data_path2 = Path(tmp_dir.name, src_dataset_path2)
                  fs.download(src_dataset_path2, data_path2.as_posix(), recursive=True)
            else:
                data_path = os.path.abspath(os.path.join(dataset_path, src))
            if 'connection_url' in value:
              ret.views_map[key]['connection_url'] =  "sqlite:///"+data_path
            else:
              ret.views_map[key]['path'] =  data_path
        return Datastore.from_dataset(ret)

if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    from datasets import load_dataset
    import fsspec, requests, aiohttp
    # some test code to manage oscar downloading
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
    if "-launch_dask_node" == args[0]:
      scheduler_file = args[1]
      DistributedContext.launch_dask_node(scheduler_file)
        
    if "-test_igzip_basic" in args:
      if not os.path.exists("wikitext-2"):
        os.system('wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip')
        os.system('unzip wikitext-2-v1.zip')
        os.system('gzip wikitext-2/wiki.train.tokens')
      vi1 = get_file_read_obj("wikitext-2/wiki.train.tokens.gz")
      assert len(vi1[[0, 5, 1000]]) == 3
      assert len(vi1[0:]) == len(vi1)
    if "-test_sql_basic" in args:
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
    if "-test_igzip" in args:
      if not os.path.exists("wikitext-2"):
        os.system('wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip')
        os.system('unzip wikitext-2-v1.zip')
        os.system('gzip wikitext-2/wiki.train.tokens')
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
        data = load_dataset("oscar", "unshuffled_deduplicated_yo")['train']
        datastore = Datastore.from_dataset(data)
        Datastore.db_connection = {}
        Datastore.db_table = {}
        datastore= datastore.move_to_sql('text')
    if "-test_load_save" in args:
        if not os.path.exists("wikitext-2"):
          os.system('wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip')
          os.system('unzip wikitext-2-v1.zip')
          os.system('gzip wikitext-2/wiki.train.tokens')
        data = load_dataset("oscar", "unshuffled_deduplicated_yo")['train']
        datastore = Datastore.from_dataset(data)
        datastore = datastore.add_mmap('embed', [-1, 512, 512], )
        Datastore.db_connection = {}
        Datastore.db_table = {}
        datastore= datastore.move_to_sql('text') 
        print (datastore)
        print (datastore[-1])
        datastore = datastore.save_to_disk("/content/test")
        print (datastore)
        print (datastore[-1])
        datastore = datastore.save_to_disk("/content/test", move_files=True)
        print (datastore)
        print (datastore[-1])
        datastore= Datastore.load_from_disk("/content/test")
        print (datastore)
        print (datastore[-1])
        datastore = datastore.add_igzip("txt", "wikitext-2/wiki.train.tokens.gz")
        datastore = datastore.save_to_disk("/content/test2", move_files=True)
    

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

""" A distributed datastore based on Huggingface's datasets and Dask"""


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

Scheduler = Union[str, Client]


logger = logging.get_logger(__name__)

	
#import IPython
#import ipysheets
#import pandas as pd
#def display_results(batch):
#    IPython.display(pd.DataFrame(batch))

####################################################################################################
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

# This is used for seeing if files from a remote drive have finished loading. Files could be in flight while we try to retreive them.
def wait_until_files_loaded(flist, max_tries=120, fs=None): # wait 2 hrs max
  if fs is None:
    fs = os
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
      size1 = fs.stat(f).st_size
      time.sleep(min(600, 1 + incr))
      incr += 1
      if fs.stat(f).st_size == size1:
        flist[i] = None
        num_done += 1
        yield f
      else:
        flist[i]=[f,incr]
    if num_done == len(flist):
      return
  return

class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    
# FileLock does not really work for gdrive or other types of shared drives (ftp). 
# So we create a SharedFileLock instead, which is not guranteed to lock a file, but the best we have.
# This is because if two nodes are trying to write to the same file, gdrive will for example create a file.txt and a file(1).txt as the other file being written to.
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


#####################################################################################################
# The hook for the distributed context over Dask with an optional streamlit app attached.  
# some of the ngrok code is based on https://github.com/gstaff/flask-ngrok which is licensed under Apache v2
#
# NOTE: This class is intended to run *one* dstributed context per node (e.g., per instance of python3 running on a macine). Thus
# several varaibles are stored in the class scope of DistributedContext. This class is *not* intended to launch a new dask worker or scheduler
# per instanance of DistributedContext.
#
# TODO: make to work without ngrok.
# TODO: streamlit does tracking. For privacy and security concern, we need to turn this off.
#
# Example:
#     distributed_context = DistributedContext()(dask_scheduler_file="<my scheduler file>", token=getpass("token:"), clear_streamlit_app=True)
# if you do not provide a custom <my app file>.py file, an app file will automatically be created, and the file name stored in DistributedContext.streamlit_app_file

class DistributedContext:

  """Distributed context over Dask to perform distributed processing, using a shared file for finding the scheduler, with an optional streamlit app attached"""

  ngrok = None
  dask_node_id = None
  dask_nodes = {}
  dask_client = None
  dask_scheduler_file = None
  streamlit_app_file = None
  streamlit_process = None

  def __init__(self, st=None, dask_scheduler_file=None, ngrok_token=None, streamlit_port=8501,  streamlit_app_file=None, hostname=None, import_code="", header_html="", num_procs=4, clear_streamlit_app=False, use_ngrok=True, flask_app=None, fs=None, time_out=43200,):
    global print
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
        print("warning: streamlit not found. Install with pip install streamlit")
    else:
      self.st = None
    DistributedContext.start(dask_scheduler_file=dask_scheduler_file, ngrok_token=ngrok_token, streamlit_port=streamlit_port,  streamlit_app_file=streamlit_app_file, hostname=hostname, import_code=import_code, header_html=header_html, num_procs=num_procs, clear_streamlit_app=clear_streamlit_app, use_ngrok=use_ngrok, flask_app=flask_app, fs=fs, time_out=time_out)
    self._dask_nodes = DistributedContext.dask_nodes

  #TODO, create a __del__ method and do an automatic .stop() after all references to DistributedContext.dask_nodes have been detached.

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
    """ log to a streamlit app """
    # TODO, write to sql log, gspread log file
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
  def reload_dask_nodes(dask_scheduler_file, time_out=43200, fs=None):
    if fs is None:
      fs = os
    with SharedFileLock(dask_scheduler_file+".lock"):
      file = dask_scheduler_file.split(".")
      if fs is not os:
        files = fs.ls(".".join(file[:len(file)-1])+"*."+file[-1])
      else:
        files = glob.glob(".".join(file[:len(file)-1])+"*."+file[-1])
      has_timeout = False
      dask_nodes=DistributedContext.dask_nodes
      for a_file in wait_until_files_loaded(files, fs):
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
              if node_id in dask_nodes:
                del dask_nodes[node_id]
              if node_id in (0, DistributedContext.dask_node_id) and DistributedContext.dask_client is not None:
                DistributedContext.dask_client.shutdown()
                DistributedContext.dask_client = None
            else:
              dask_nodes[node_id] = (address, last_heartbeat) # what about the case where a node removes itself?
      if has_timeout:
        with fs.open(dask_scheduler_file, "w") as f: 
          for key, val in DistributedContext.dask_nodes.items():
            f.write(str(key)+"\t"+str(val[0])+"\t"+str(val[1])+"\n")
      if fs is not os:
        files = fs.ls(".".join(file[:len(file)-1])+"*."+file[-1])
      else:
        files = glob.glob(".".join(file[:len(file)-1])+"*."+file[-1])
      if DistributedContext.dask_node_id == 0 and len(files) > 1:
        # let's do a sanity check to see if there are more than one dask_scheduler_file
        # this might happen because locking is not perfect over shared drives. 
        # if there are duplicates, we will merge the files

        for a_file in wait_until_files_loaded(files):
          if fs.path.exists(a_file):
            fs.unlink(a_file) 
        with fs.open(dask_scheduler_file, "w") as f: 
          for key, val in DistributedContext.dask_nodes.items():
            f.write(str(key)+"\t"+str(val[0])+"\t"+str(val[1])+"\n")

  # todo, launch flask instead of streamlit, run in no_grok mode.
  @staticmethod
  def start(dask_scheduler_file=None, ngrok_token=None, streamlit_port=8501,  streamlit_app_file=None, hostname=None, import_code="", header_html="", num_procs=4, clear_streamlit_app=False, use_ngrok=True, flask_app=None, fs=None, time_out=43200,):
    if fs is None:
      fs = os
    if ngrok_token:
      DistributedContext.write_authtoken(ngrok_token)

    if dask_scheduler_file is None:
      dask_scheduler_file = DistributedContext.dask_scheduler_file
    if DistributedContext.dask_scheduler_file is not None and dask_scheduler_file != DistributedContext.dask_scheduler_file:
      raise RuntimeError(f"dask_scheduler_file already set. expected {DistributedContext.dask_scheduler_file}")
    DistributedContext.dask_scheduler_file = dask_scheduler_file
    if dask_scheduler_file is None:
        raise RuntimeError("no dask_scheduler_file provided")
    DistributedContext.reload_dask_nodes(dask_scheduler_file, time_out=time_out)
    
    generic_streamlit_app = False
    if streamlit_app_file is None:
      generic_streamlit_app = True
      streamlit_app_file= str(Path(tempfile.gettempdir(), "app.py"))
    DistributedContext.streamlit_app_file = streamlit_app_file

    with SharedFileLock(dask_scheduler_file+".lock", fs=fs):
      if not DistributedContext.dask_nodes or 0 not in DistributedContext.dask_nodes:
        try:
          DistributedContext.dask_node_id = 0
          DistributedContext.dask_process = subprocess.Popen(["dask-scheduler", address, '--name', f"worker_{DistributedContext.dask_node_id}", "--nprocs", num_proces, "--nthreads", "1", "--no-dashboard"], text=True, capture_output=True, stderr=subprocess.STDOUT, preexec_fn = preexec) 
          atexit.register(DistributedContext.dask_process.terminate)
          time.sleep(5)
          dask_port = int(DistributedContext.dask_process.stdout.split(":")[-1].split()[0].strip())
          DistributedContext.dask_client = Client(dask_port)
          addresses = DistributedContext._run_ngrok(streamlit_port, tcp_port=dask_port, hostname=hostname, region="us")
          DistributedContext.dask_nodes[DistributedContext.dask_node_id] = ([a for a in addresses if a.startswith("tcp:")][0], datetime.timestamp(datetime.now()))
          print ("dask schduler running on", DistributedContext.dask_nodes[DistributedContext.dask_node_id][0])
        except e:
          DistributedContext.stop()
          raise e
      else:
        try:
          address = DistributedContext.dask_nodes[0][0]
          DistributedContext.dask_node_id = max(list(DistributedContext.dask_nodes.keys()))
          # there is an error in connecting to the scheduler, then the scheduler has probably died and we need to create a new scheduler with this node.
          DistributedContext.dask_client = Client(DistributedContext.dask_nodes[0][0])
          DistributedContext.dask_process = subprocess.Popen(["dask-worker", address, '--name', f"worker_{DistributedContext.dask_node_id}", "--nprocs", num_proces, "--nthreads", "1", "--no-dashboard"], text=True, capture_output=True, stderr=subprocess.STDOUT, preexec_fn = preexec) 
          atexit.register(DistributedContext.dask_process.terminate)
          DistributedContext.dask_nodes[DistributedContext.dask_node_id] = (DistributedContext.dask_node_id, datetime.timestamp(datetime.now()))
          DistributedContext.reload_dask_nodes(dask_scheduler_file, time_out=time_out)
        except e:
          DistributedContext.stop()
          raise e
      with open(dask_scheduler_file, "w") as f: 
        for key, val in DistributedContext.dask_nodes.items():
          f.write(str(key)+"\t"+str(val[0])+"\t"+str(val[1])+"\n")
    webaddr = [a for a in addresses if a.startswith("https:")][0]
    if generic_streamlit_app and (not os.path.exists(streamlit_app_file) or clear_streamlit_app):
      DistributedContext.create_streamlit_app(streamlit_app_file, import_code=import_code, header_html=header_html)
    dir, filename = os.path.split(streamlit_app_file)
    cwd = os.getcwd()
    os.chdir(dir)
    DistributedContext.streamlit_process = subprocess.Popen(('streamlit', 'run', streamlit_app_file), preexec_fn = preexec)
    os.chdir(cwd)
    atexit.register(DistributedContext.streamlit_process.terminate)
    print (f"streamlit server running on {webaddr}")
    DistributedContext.reload_dask_nodes(dask_scheduler_file, time_out=time_out)
    return DistributedContext.dask_client 

  @staticmethod
  def stop():
    if DistributedContext.ngrok is not None:
      DistributedContext.ngrok.terminate()
    DistributedContext.ngrok = None
    if DistributedContext.streamlit_process is not None:
      DistributedContext.streamlit_process.terminate()
    DistributedContext.streamlit_process = None
    if DistributedContext.dask_process is not None:
      DistributedContext.dask_process.terminate()
    DistributedContext.dask_process = None
    if DistributedContext.dask_client is not None:  
      DistributedContext.dask_client.shutdown()
    DistributedContext.dask_client = None
    DistributedContext.dask_node_id = None
    DistributedContext.dask_nodes.clear()
    DistributedContext.dask_scheduler_file = None
    
  # used to test the postgresd db interconnect. TODO, launch an ngrok tcp tunnel to the database_port
  @staticmethod
  def start_postgress_test_db(username, password):
    if not os.environ.get("POSTGRESS_DATABASE_HOST"):
      # Install postgresql server
      os.system("sudo apt-get -y -qq update")
      os.system("sudo apt-get -y -qq install postgresql")
      os.system("sudo service postgresql start")

      # Setup a password and username
      os.system(f'sudo -u postgres psql -U postgres -c "ALTER USER {username} PASSWORD \'{password}\';"')
      os.environ["POSTGRESS_DATABASE_NAME"] = "test"
      os.environ["POSTGRESS_DATABASE_HOST"] = "localhost"
      os.environ["POSTGRESS_DATABASE_PORT"] ="5432"
      os.environ["POSTGRESS_DATABASE_USER"] = username
      os.environ["POSTGRESS_DATABASE_PASS"] = password

    postgress_connection_url="postgresql://{}:{}@{}?port={}&dbname={}".format(
        os.environ['POSTGRESS_DATABASE_USER'],
        os.environ['POSTGRESS_DATABASE_PASS'],
        os.environ['POSTGRESS_DATABASE_HOST'],
        os.environ['POSTGRESS_DATABASE_PORT'],
        os.environ['POSTGRESS_DATABASE_NAME'],
    )

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
    if DistributedContext.ngrok is not None:
      raise RuntimeError("ngrok is already running")
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

######################################################################################
# SqlAlchemy/Dataset based classes for interconnect with SQL, and in particular sqlite3
### A note about naming: dataset and datasets are not the same libraries.

def multi_iter_result_proxy(rps, step=None, batch_fn=None, row_type=row_type):
    """Iterate over the ResultProxy."""
    for rp in rps:
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


class ResultIterExt(object):
    """SQLAlchemy ResultProxies are not iterable to get a
    list of dictionaries. This is to wrap them."""

    def __init__(self, result_proxy, step=None, row_type=row_type, batch_fn=None):
        self.row_type = row_type
        if type(result_proxy) is not list:
            result_proxy = [result_proxy]
        self.result_proxy = result_proxy
        self.batch_fn = batch_fn
        try:
            self.keys = list(result_proxy[0].keys())
            #self.keys.append('rank')
            #print (self.keys)
            self._iter = multi_iter_result_proxy(result_proxy, step=step, batch_fn=self.batch_fn, row_type=self.row_type)
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



class TableExt(dataset.Table):
    """ Extends dataset.Table's functionality to work with Datastore(s) and to add full-text-search (FTS) """


    def __init__(
        self,
        database,
        table_name,
        primary_id=None,
        primary_type=None,
        primary_increment=None,
        auto_create=False,
    ):
      super().__init__(database, table_name, primary_id=primary_id, primary_type=primary_type, primary_increment=primary_increment, auto_create=auto_create)
      self.external_fts_columns = {}
      self.has_fts_trigger = False

    def find(self, *_clauses, **kwargs):
        """Perform a search on the table similar to dataset.Table's
        find, except: optionally gets a result only for specific
        columns by passing in _columns keyword. And optionally perform
        FTS on a Sqlite3 table.
        
        
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

        _fts_query = kwargs.pop('_fts_query', None)
        _fts_step = kwargs.pop('_fts_step', QUERY_STEP)
        _columns = kwargs.pop('_columns', None)
        _limit = kwargs.pop('_limit', None)
        _offset = kwargs.pop('_offset', 0)
        order_by = kwargs.pop('order_by', None)
        _streamed = kwargs.pop('_streamed', False)
        _step = kwargs.pop('_step', QUERY_STEP)
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
              if self.name.endswith(f"_fts_idx"):
                if not self.name.endswith(f"_{column}_fts_idx"):
                   raise RuntimeError(f"table  {self.name} does not match expected table name for column {column}")
                db, table_name = self.db, self.name
                return_id2rank_only = True
              else:
                db, fts_table_name = self.db, f"{self.name}_{column}_fts_idx"
              # TODO: check if sqlite3 
            q = q.replace("'","\'") 
            if _limit is not None:
              # using _limit is a hack. we want a big enough result so that subsequent query will narrow down to a good result at _limit,
              # but we don't want all results which could be huge. we can set the fts_max_limit as a parameter to the table or db level.
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
        #we do an optimization here by just returning the id, if there are no queries and the return _columns = ['id']
        if id2rank and ((_columns and len(_columns) and self._primary_id in _columns) or return_id2rank_only):
          ret = list(id2rank.items())
          ret.sort(key=lambda a: a[0])
          return [a[1] for a in ret]
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
        if fts_results:
          len_fts_results = len(fts_results)
          for rng in range(0, len_fts_results, _fts_step):
            max_rng = min(len_fts_results, rng+_fts_step)
            kwargs[self._primary_id] ={'in': [int(a[1]) for a in fts_results[rng:max_rng]]}
            args = self._args_to_clause(kwargs, clauses=_clauses)

            if _columns is None:
                query = self.table.select(whereclause=args,
                                      limit=_limit,
                                      offset=_offset)
            else:
                query = self.table.select(whereclause=args,
                                      limit=_limit,
                                      offset=_offset).with_only_columns([self.table.c[col] for col in _columns])
            results.append(conn.execute(query))
            if len(order_by):
                query = query.order_by(*order_by)
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
        else:
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
          results=[conn.execute(query)]
        return ResultIterExt(results,
                          row_type=row_type_fn,
                          step=_step, batch_fn=batch_fn)


    def update(self, row, keys, ensure=None, types=None, return_count=False):
      row = self._sync_columns(row, ensure, types=types)
      args, row = self._keys_to_args(row, keys)
      old = list(self.find(**args))
      ret =  super().update(row, keys=keys, ensure=ensure, types=types, return_count=return_count)
      if old:
        for key in row.keys():
          self.update_fts(column=key, old_data = old, new_data=row, mode='update')
      return ret

    def update_many(self, rows, keys, chunk_size=1000, ensure=None, types=None):
      row = self._sync_columns(row, ensure, types=types)
      args, row = self._keys_to_args(rows, keys) # this probably won't work
      old = list(self.find(**args))
      ret =  super().update_many(rows, keys=keys, chunk_size=chunk_size, ensure=ensure, types=types)
      if old:
        for key in rows[0].keys():
          self.update_fts(column=key, old_data = old, new_data=rows, mode='update')
      return ret

        
    def delete(self, *clauses, **filters):
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

    # sqlite3 fts manual updates. we create TableExt level updates when we don't have actual triggers in the sqlite3 database.
    def update_fts(self, column, new_data=None, old_data=None, mode='insert'):
      if column != self._primary_id and column in self.external_fts_columns:
        db, fts_table_name = self.external_fts_columns[column]
        db.update_fts(fts_table_name, column=column, new_data=new_data, old_data=old_data, mode=mode, primary_id=self._primary_id)
      

class DatabaseExt(dataset.Database):
    """A DatabaseExt textends dataset.Database and represents a SQL database with multiple tables of type TableExt."""
    
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

######################################################################################
# Indexed Gzip - code for accessing for efficiently gzip files.

class IndexGzipFileExtBlocks:
    """An indexable structure made of multiple gzip files that are
    viewed as a single line indexed file. Each igzip is lazy loaded
    when needed.
    :arg blocks: A list of dict, with fields 'file', 'start', 'end' and optionaly 'fobj'.
    :arg mode: either 'r' or 'rb
    :arg fs: the fsspec file system or None for local os file system.
    :arg chunksize: the chunk size to use when iterating.
    """
    def __init__(self, blocks, mode="rb", fs=None, chunksize=1000):
      self.blocks = blocks
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
        return max(block[-1] for block in self.blocks)

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
          for block in self.blocks:
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
          for block in self.blocks:
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


#################################################################################################################################
# Code for Snorkel label function appliers which for Datastores. 

class DatastoreLFApplier(BaseLFApplier):
    """LF applier for a Datastore DataFrame.  Datastore DataFrames
    consist of Dask dataframes in certain columns, and just-in-time
    sharded loading of data from SQL or shared drive in other columns.
    We can convert everything to pandas and use the snorkel underlying
    pandas LF.  This allows for efficient parallel computation over
    DataFrame rows.  For more information, see
    https://docs.dask.org/en/stable/dataframe.html
    """

    def apply(
        self,
        df: "Datastore",
        scheduler: Scheduler = "processes",
        fault_tolerant: bool = False,
    ) -> np.ndarray:
        """Label Datastore of data points with LFs.
        Parameters
        ----------
        df
            Datastore containing data points to be labeled by LFs
        scheduler
            A Dask scheduling configuration: either a string option or
            a ``Client``. For more information, see
            https://docs.dask.org/en/stable/scheduling.html#
        fault_tolerant
            Output ``-1`` if LF execution fails?
        Returns
        -------
        np.ndarray
            Matrix of labels emitted by LFs
        """
        f_caller = _FunctionCaller(fault_tolerant)
        apply_fn = partial(apply_lfs_to_data_point, lfs=self._lfs, f_caller=f_caller)
        labels = df.map(lambda p_df: p_df.apply(apply_fn, axis=1), distributed_context=scheduler)
        labels_with_index = rows_to_triplets(labels)
        return self._numpy_from_row_data(labels_with_index)

class DatastoreSFApplier(DatastoreLFApplier):  # pragma: no cover
    """SF applier for a Datastore DataFrame.
    """

    _use_recarray = True


# TODO, all LF applications will write to a column, either mmemmap in original dataset or a new dataset.

#################################################################################################################################
# Code for megatron mmap indexed datasets, which are in turn based on memmap datasets, and a custom numpy based index file.


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

######################################################################################
# Main extensions to Huggingface's datasets to create a datastore that interconnects to many backends
# and supports clustered storage and processing.
#
### We want to have mutliple types of storage that ideally can be
### transported as a file transfer with an arrow dataset (perhaps a tar file?). So if we
### have <signature>.arrow, we may have fts_<signature>.db (for full
### text indexing sqlite database) and <signature>.db (sqlite database), and
### <siganture>.mmap (mmap file reprsenting a tensor), and
### <singature>.igz (if we wish to store some portion of the text
### columns in igzip format for compression and legacy purposes.

#NOTE: datasets uses the terms 'features' and 'columns' interchangably.
class FeaturesWithViews(Features):
    """ an extension of Features that allows printing of the views as well """
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
        table: TableExt = None,
        column: Optional[str] = None
    ):
        self.table=table
        self.column=column
        
    def search(self, query, k: int = 10) -> SearchResults:
        hits= list(self.table.find(_fts_query=[(self.column, query)], _limit=k))
        return SearchResults([hit["rank"] for hit in hits], [int(hit["rowid"]) for hit in hits])
        
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
    
class Datastore(Dataset): 
    """
    A class that wraps a Huggingface arrow based Dataset to provide
    some distributed processing over Dask and optimized reading and
    *writing* in various persistance backends.  

    Currently provides support for features bound to memmap, sharded
    indexed gzip (igzip) file, and sqlalchemy databases.
    
    Also permits full text indexing and searching (via .filter or
    .search) into a sqlite database for any text feature in a dataset.
    """
        
    def __repr__(self):
        ret = FeaturesWithViews(self._info.features)
        ret.views_map = {} if not hasattr(self, "views_map") else self.views_map
        return f"Datastore({{\n    features: {ret},\n    num_rows: {self.num_rows}\n}})"
    

    @classmethod
    def from_dataset(cls, dataset, template_datastore=None, views_map=None, id_feature=None, pipelines_manager=None, id2idx_identity=None,):
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

        if  hasattr(dataset, "pipelines_manager"):
          self.pipelines_manager = dataset.pipelines_manager
        elif  pipelines_manager is not None:
          self.id2idx_identity = pipelines_manager
        elif hasattr(template_datastore, "pipelines_manager"):
          self.pipelines_manager = template_datastore.pipelines_manager
        else:
          self.pipelines_manager = None

        if  hasattr(dataset, "id_feature"):
          self.id_feature = dataset.id_feature
        elif  id_feature is not None:
          self.id_feature = id_feature
        elif hasattr(template_datastore, "id_feature"):
          self.id_feature = template_datastore.id_feature
        else:
          self.id_feature = "id"

        if  hasattr(dataset, "views_map"):
          self.views_map = copy.deepcopy(dataset.views_map)
        elif  views_map is not None:
          self.views_map = copy.deepcopy(views_map)
        elif hasattr(template_datastore, "views_map"):
          self.views_map = copy.deepcopy(template_datastore.views_map)
        else:
          self.views_map = {}

        return self

    #NOTE:if you remove a field that was previously monitored, the metadata generated from it will not be removed too.
    def apply_pipelines_manager(self, pipelines_manager=None, batch_size=1000, num_proc=4, ):
      if hasattr(self, 'pipelines_manager') and self.pipelines_manager not in (None, pipelines_manager):
          print(f"warning: resetting the metadta_manager to {pipelines_manager}")
      if pipelines_manager is not None:
          self.pipelines_manager = pipelines_manager
      if self.pipelines_manager is not None and self.pipelines_manager.preprocess:
        self = self.map(self.pipelines_manager.preprocess,  batch_size=batch_size, batched=True, num_proc=num_proc)
      if self.pipelines_manager is not None and self.pipelines_manager.postprocess:
        self = self.map(self.pipelines_manager.postprocess,  batch_size=batch_size, batched=True, num_proc=num_proc)
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

    # we use class variables because we don't want it serialized in an instance of this dataset. this might take up too much memory, so we might use an LRU cache to clear things out.
    igzip_fobj = {}
    def _get_igzip_fobj(self, feature, start=None, end=None):
        if feature in self.views_map and self.views_map[feature]['type'] == 'igzip':
          files = self.views_map[feature]['path']
          if type(files) is str:
            file_path = files
            if file_path in Datastore.igzip_fobj:
              fobj = Datastore.igzip_fobj[file_path]
            else:
              Datastore.igzip_fobj[file_path] = fobj = IndexGzipFileExt()
            return fobj
          else:
            file_path = tuple(file['file'] for file in files)
            if file_path in Datastore.igzip_fobj:
              fobj = Datastore.igzip_fobj[file_path]
            else:
              Datastore.igzip_fobj[file_path] = fobj = IndexGzipFileExtBlocks(files)
            return fobj
        else:
          raise RuntimeError(f"{feature} is not a igzip type")

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
    def from_mmap(cls,  feature_view, shape, path=None, dtype='float32', dtype_str_len=100, id_feature="id", batch_size=1000, num_proc=4, pipelines_manager=None):
      return cls.from_dict({}).add_mmap(feature_view=feature_view, shape=shape, path=path, dtype=dtype, dtype_str_len=dtype_str_len, id_feature=id_feature, batch_size=batch_size, num_proc=num_proc, pipelines_manager=pipelines_manager)


    def move_to_mmap(self, src_feature, dst_feature_view=None, shape=None, path=None, dtype='float32', dtype_str_len=100, id_feature="id", batch_size=1000, num_proc=4, pipelines_manager=None):
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
      self.add_mmap(feature_view=dst_feature_view, shape=shape, path=path, dtype=dtype, id_feature=id_feature, batch_size=batch_size, num_proc=num_proc) #don't pass in the pipelines_manager
      val = self.views_map[dst_feature_view]
      self.map(Datastore._move_to_mmap_col, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'src_feature':src_feature, 'id_feature':id_feature, 'path': val['path'], 'dtype': val['dtype'], 'shape':shape})
      self= self.remove_columns(src_feature)
      if hasattr(self, 'pipelines_manager') and self.pipelines_manager not in (None, pipelines_manager):
          print(f"warning: resetting the metadta_manager to {pipelines_manager}")
      if pipelines_manager is not None:
          self.pipelines_manager = pipelines_manager
      #only apply the pipelines_manager if we are moving to a new feature and that feature is monitored. 
      if pipelines_manager is not None and src_feature.startswith ("__tmp__") and  dst_feature_view in pipelines_manager.features_monitored() and self.pipelines_manager.postprocess:
        self = self.map(pipelines_manager.postprocess,  batch_size=batch_size, batched=True, num_proc=num_proc)
      return self

    def add_mmap(self, feature_view, shape, path=None, dtype='float32', dtype_str_len=100, id_feature="id", batch_size=1000, num_proc=4, pipelines_manager=None):
      """"mapping a feature/columun to a memmap array accessed by row"""
      if not hasattr(self, 'views_map'): self.views_map = {}
      if hasattr(self, 'id_feature') and self.id_feature != id_feature:
        raise RuntimeError(f"attempting to reset the index to {id_feature}")
      else:
        self.id_feature = id_feature
      if hasattr(self, 'pipelines_manager') and self.pipelines_manager not in (None, pipelines_manager):
          print(f"warning: resetting the metadta_manager to {pipelines_manager}")
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
      if id_feature not in self.features:
        if len(self) == 0 and shape[0] > 0:
            self = Datastore.from_dataset(Dataset.from_dict({id_feature: range(shape[0])}), self)
            ids = dict([(a,1) for a in range(len(self))])
            self.id2idx_identity = True
        else:
            self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'id_feature': id_feature})
            ids = dict([(a,1) for a in range(len(self))])
            self.id2idx_identity = True
      else:
        ids = dict([(a,1) for a in self[id_feature]])
      missing_ids = []
      for id in range(shape[0]):
          if id not in ids:
            missing_ids.append(id)
      if missing_ids:
            self = self.add_batch({id_feature: missing_ids})
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


    @classmethod
    def from_igzip(cls, feature_view, path,  id_feature="id", batch_size=1000, num_proc=4, pipelines_manager=None):
      return cls.from_dict({}).add_igzip(feature_view=feature_view, path=path,  id_feature=id_feature, batch_size=batch_size, num_proc=num_proc, pipelines_manager=pipelines_manager)


    def add_igzip(self, feature_view, path,  id_feature="id", batch_size=1000, num_proc=4, pipelines_manager=None, add_fts=False, fts_connection=None, fts_table_name=None):
      """    
      mapping a feature/columun to an indexed gzip file accessed by line 
      """
      if not hasattr(self, 'views_map'): self.views_map = {}
      if hasattr(self, 'id_feature') and self.id_feature != id_feature:
        raise RuntimeError(f"attempting to reset the index to {id_feature}")
      else:
        self.id_feature = id_feature
      if hasattr(self, 'pipelines_manager') and self.pipelines_manager not in (None, pipelines_manager):
          print(f"warning: resetting the metadta_manager to {pipelines_manager}")
      if pipelines_manager is not None:
          self.pipelines_manager = pipelines_manager
      fobj = self._get_igzip_fobj(path)
      if id_feature not in self.features:
          if len(self) == 0:
            self = Datastore.from_dataset(Dataset.from_dict({id_feature: range(len(fobj))}), self)
            ids = dict([(a,1) for a in range(len(self))])
            self.id2idx_identity = True
          else:
            print ("adding idx")
            self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'id_feature': id_feature})
            ids = dict([(a,1) for a in range(len(self))])
            self.id2idx_identity = True
      else:
          ids = dict([(a,1) for a in self[id_feature]])
      missing_ids=[]
      for id in range(len(fobj)):
            if id not in ids:
              missing_ids.append(id)
      if missing_ids:
              self = self.add_batch({id_feature: missing_ids})
              if not hasattr(self, 'id2idx_identity'):  self.id2idx_identity = True
              if self.id2idx_identity:
                contiguous, start, end = is_contiguous(missing_ids)
                self.id2idx_identity = start ==len(self) and contiguous
              else:
                self.id2idx_identity = False
      if fts_table_name is None:
        fts_table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}_{column}_fts_idx"
      if not fts_connection:
        fts_connection="sqlite:///"+self.cache_files[0]['filename'].replace(".arrow", ".db")
      self.views_map[feature_view] = {'type':"igzip", 'path': path, 'fts': add_fts, 'fts_db': fts_connection, 'fts_table_name': fts_table_name}
      if pipelines_manager is not None and feature_view in pipelines_manager.features_monitored() and self.pipelines_manager.postprocess:
        self = self.map(pipelines_manager.postprocess,  batch_size=batch_size, batched=True, num_proc=num_proc)
      return self


    def move_to_sql(self, src_feature_to_dst_views_map, table_name=None, connection_url=None,  id_feature="id",  batch_size=1000, num_proc=4, pipelines_manager=None):
      if table_name is None:
          #print (self.info.builder_name, self.info.config_name)
          table_name = f"_{self._fingerprint}_{self.info.builder_name}_{self.info.config_name}_{self.split}"
      if not connection_url:
          connection_url="sqlite:///"+self.cache_files[0]['filename'].replace(".arrow", ".db")
      table = Datastore._get_db_table(self, table_name, connection_url)
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
      self.add_sql(feature_view=feature_view, table_name=table_name, connection_url=connection_url, id_feature=id_feature, batch_size=batch_size, num_proc=num_proc)
      self = self.map(Datastore.upsert_sql_from_batch, batch_size=batch_size, batched=True, num_proc=1 if connection_url=="sqlite://" else num_proc, fn_kwargs={'views_map':self.views_map, 'id_feature':id_feature, 'src_feature_to_dst_views_map': src_feature_to_dst_views_map})
      self = self.remove_columns(src_feature)
      if hasattr(self, 'pipelines_manager') and self.pipelines_manager not in (None, pipelines_manager):
          print(f"warning: resetting the metadta_manager to {pipelines_manager}")
      if pipelines_manager is not None:
          self.pipelines_manager = pipelines_manager
      if pipelines_manager is not None and self.pipelines_manager.postprocess:
        self = self.map(pipelines_manager.postprocess,  batch_size=batch_size, batched=True, num_proc=num_proc)
      return self

    @classmethod
    def from_sql(cls,  feature_view, table_name, connection_url, dtype="str", id_feature="id",  batch_size=1000, num_proc=4, pipelines_manager=None):
      return cls.from_dict({}).add_sql(feature_view=feature_view, table_name=table_name, connection_url=connection_url, dtype=dtype, id_feature=id_feature, batch_size=batch_size, num_proc=num_proc, pipelines_manager=pipelines_manager)

    def add_sql(self, feature_view=None, table_name=None, connection_url=None, dtype="str", id_feature="id",  batch_size=1000, num_proc=4, pipelines_manager=None):
        """
        mapping one or more columns/features to a sql database. creates a sqlalchmey/dataset dynamically with id_feature as the primary key. 
        TODO: remember to strip passwords from any connection_url. passwords should be passed as vargs and added to the conneciton url dynamically
        passwords should not be perisisted.
        NOTE: this dataset will not automatically change if the database changes, and vice versa. periodically call this method again to sync the two or create callbacks/triggers in your code.
        """
        if not hasattr(self, 'views_map'): self.views_map = {}
        if hasattr(self, 'id_feature') and self.id_feature != id_feature:
          raise RuntimeError(f"attempting to reset the index to {id_feature}")
        else:
          self.id_feature = id_feature
        if hasattr(self, 'pipelines_manager') and self.pipelines_manager not in (None, pipelines_manager):
          print(f"warning: resetting  the metadta_manager to {pipelines_manager}")
        if pipelines_manager is not None:
          self.pipelines_manager = pipelines_manager
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
            self.id2idx_identity = True
          else:
            self = self.map(Datastore._add_idx, with_indices=True, batch_size=batch_size, batched=True, num_proc=num_proc, fn_kwargs={'id': id_feature})
            ids = dict([(a,1) for a in range(len(self))])
            self.id2idx_identity = True
        else:
          ids = dict([(a,1) for a in self[id_feature]])
        missing_ids = []
        for id in table_ids:
          if id[id_feature] not in ids:
            missing_ids.append(id[id_feature])
        if missing_ids:
            self = self.add_batch({id_feature: missing_ids})
            if not hasattr(self, 'id2idx_identity'):  self.id2idx_identity = True
            if self.id2idx_identity:
              contiguous, start, end = is_contiguous(missing_ids)
              self.id2idx_identity = start ==len(self) and contiguous
            else:
              self.id2idx_identity = False
        do_pipelines_manager = False
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
            if pipelines_manager is not None and cl in pipelines_manager.features_monitored():
              do_pipelines_manager = True
        if do_pipelines_manager and self.pipelines_manager.postprocess:
          self = self.map(pipelines_manager.postprocess,  batch_size=batch_size, batched=True, num_proc=num_proc)
        return self
    
    def add_sqlite3_fts_index(self, feature_view=None, table_name=None, connection_url=None, dtype="str", id_feature="id",  batch_size=1000, num_proc=4, pipelines_manager=None):
      pass

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
      "find" method from dataset.Table.  example:
      dataset.filter(sql_query={'lang':'ru'}) will return those items
      in the dataset that has the language 'ru'.
      """
      if not hasattr(self, 'views_map'): self.views_map = {}
      if sql_query or fts_query:
        if not sql_query:
          sql_query = {}
        if fts_query:
          sql_query['_fts_query'] = fts_query
        found_table = None
        for key, item in sql_query.item():
          found = False
          for feature_view, val in self.views_map.items():
            if val['type']=='sql':
              table = self._get_db_table(val['table_name'], val['connection_url'])
              if key in table.columns:
                if found_table and table != found_table:
                  raise RuntimeError("filtering from multiple sql tables at the same time not supported. do filtering in sequence instead.")
                found_table = table
                found = True
                break
          if not found:
            raise RuntimeError(f"query on a column {key} that is not a sql column")
        sql_query['_columns'] = [self.id_feature]
        ids = dict([(val['id'],1) for val in found_table.find(*[], **sql_query)])
        if hasattr(self, 'id2idx_identity') and self.id2idx_identity:
          ret = self.select(ids)
          ret.id2idx_identity=False
        else:
          ret = self
          if function:
            function = lambda example: example['id'] in ids and function(example) 
          else:
            function = lambda example: example['id'] in ids
      else:
        ret = self
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
      return self.map(
          partial(map_function, function=function, with_indices=with_indices),
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

    # note that while the id_feature corresponds to an item in an external storage, accessing an arrow dataset by datataset[index]
    # will not be guranteed to get the corresponding id. a[0] will return the first item in the current subset of the dataset. 
    # but a[0] does not necessarily return {'id':0, ...}
    # instead, a[0] might return {'id': 10, 'mmap_embed': <array correponding to the 10th location in the mmap file>}. 
    # To get dataset items by 'id', use either filter or filter_sql.
    # or check the property id2idx_identity to determine if the id corresponds to the index of the table.
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
                keys = outputs_or_keys[self.id_feature]
                contiguous, start, end = is_contiguous(keys)
            else:
                raise RuntimeError("got unknown outputs or keys type")
            if outputs is None:
                outputs = df.DataFrame()
            outputs = getitems(self, outputs,  keys, contiguous, start, end, format_columns, output_all_columns, mmap_by_items=True)
            if self.id_feature in outputs and format_columns and self.id_feature not in format_columns: 
              outputs.drop(self.id_feature, axis=1) 
            if len(format_columns) == 1:
              outputs = outputs[format_columns[0]]
            return outputs
        raise RuntimeError("got unknown outputs or keys type")


  # Basic map, sort and then reduce functions over Dask using
  # Datastore as the primary data storage and multi-processer.  The
  # main file transfer and sharing are through a shared directory
  # (e.g., Google Colab) as opposed to through Dask.  Dask is used for
  # coordination of processing only.  Requires unix like programs,
  # split, cat, sort, gzip and gunzip

    @staticmethod
    def sort_merge(batch_idx_files, output_igzip_file, cache_dir=".", lock=True):
      if lock:
        lock - FileLock(output_igzip_file+".lock")
      else:
        lock = DummyLock()
      with lock:
        batch_idx_files = list(wait_until_files_loaded(batch_idx_files))
        gzipped_output = [o for o in batch_idx_files if o.endswith(".gz")]
        if gzipped_output:
          zcat = "zcat "+" < (zcat ".join(gzipped_output)
          files = " ".join([o for o in batch_idx_files if not o.endswith(".gz")])
          os.system(f"sort --parallel=32 -T {cache_dir} -n -m {files} -o {output_igzip_file} < ({zcat})")
        else:
          files
          os.system(f"sort --parallel=32 -T {cache_dir} -n -m {files} -o {output_igzip_file}")
      
    @staticmethod
    def cat(batch_idx_files, output_igzip_file, cache_dir=".", lock=True):
      if lock:
        lock - FileLock(output_igzip_file+".lock")
      else:
        lock = DummyLock()
      with lock:
        batch_idx_files = list(wait_until_files_loaded(batch_idx_files))
        gzipped_output = [o for o in batch_idx_files if o.endswith(".gz")]
        if gzipped_output:
          os.system("cat " + " ".join([o for o in batch_idx_files if not o.endswith(".gz")]) + " < (zcat "+ ") < (zcat ".join(gzipped_output) + ")"  + " > " +  cache_dir+"___tmp___" + output_igzip_file)
        else:
          os.system("cat " + " ".join(batch_idx_files) + " > " +  cache_dir+"___tmp___" + output_igzip_file)
        next(wait_until_files_loaded(cache_dir+"/___tmp___" + output_igzip_file))
        os.system("mv "+cache_dir+"/___tmp___" + output_igzip_file + " " + output_igzip_file)
      
    @staticmethod
    def sort_file(f, cache_dir=".", gzip_output=False, lock=True):
      if lock:
        lock - FileLock(f+".lock")
      else:
        lock = DummyLock()
      with lock:
        if os.path.exists(f):
          f = next(wait_until_files_loaded(f))
          os.system("sort --parallel=32 -T "+cache_dir+" -n "+f+" -o "+f)  
          if gzip_output:
            os.system(f"gzip {f}")
            return f+".gz"

    @staticmethod
    def merge_and_save_files(batch_idx_files, output_igzip_file, sort=False, shared_dir=None, gzip_output=None, split_lines=5000000, lock=True):
      """ If sorting, assume all batch_idx_files are already sorted. """
      #if the files are all on the shared dir, then move it to cache_dir
      if lock:
        lock - FileLock(f+".lock")
      else:
        lock = DummyLock()
      with lock:
        batch_idx_files = list(wait_until_files_loaded(batch_idx_files))
        batch_idx_files.sort()
        if sort:
          MapReduceNode.sort_merge(batch_idx_files, output_igzip_file, lock=False)
        else:
          MapReduceNode.cat(batch_idx_files, output_igzip_file, lock=False)
        next(wait_until_files_loaded(output_igzip_file))
        for f in batch_idx_files:
          os.unlink(f)
        output_igzip_files = []
        if os.stat(output_igzip_file).st_size > self.small_file and split_lines > 0:
          output_igzip_file0 = output_igzip_file.split(".")
          suff = output_igzip_file0[-1]
          output_igzip_file0 = ".".join(output_igzip_file0[:len(output_igzip_file0)-1])
          split_lines = max(10000, split_lines)
          os.system(f"split -l {split_lines} {output_igzip_file} {output_igzip_file0}")
          for f in glob.glob(output_igzip_file0+"*"):
            if gzip_output:
              next(wait_until_files_loaded(f))
              os.system(f"gzip -S {suff}.gz {f}")
              f = f+f"{suff}.gz"
              next(wait_until_files_loaded(f)) 
            if shared_dir:
              shutil.move(f, self.shared_dir)
              ouptut_files.append(Path(shared_dir, Path(f).name))
            else:
              ouptut_files.append(f)
        else:
          f = output_igzip_file
          if gzip_output:
            next(wait_until_files_loaded(f))
            os.system(f"gzip {f}")
            f = f+".gz"
            next(wait_until_files_loaded(f)) 
          if shared_dir:
            shutil.move(f, self.shared_dir)
            ouptut_files.append(Path(shared_dir, Path(f).name))
          else:
            ouptut_files.append(f)
        return ouptut_files
    
    @staticmethod
    def _distributed_map(dataset_path: str=None,
          shard: Tuple[int]=None,
          function: Callable = None,
          with_indices: bool = False,
          input_columns: Optional[Union[str, List[str]]] = None,
          batched: bool = True,
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
          desc: Optional[str] = None,
          curr_task_subfile_path: str =None, 
          intermediate_sort: bool =None):
        args['inputfile'] = dataset_path
        args['outfile'] = curr_task_subfile_path
        # if input_file_path sits on the shared_dir, copy to cache_dir
        datastore = Datastore.load_from_disk(dataset_path, shared_dir=shared_dir, cache_dir=cache_dir)
        # make sure we are not going to recursively send the job
        # through the distributed context.  batch should always be
        # true here, and batch size should produce together about 1gb
        # per num_proc.  we will be sending this file to a shared
        # directory every cycle. num process should be some reasonable
        # number.  if each dask node runs 4 main processes, and each 4
        # main processes runs 4 sub processes, we have 16 processes
        # running per node
        # 
        datastore.distributed_context = None
        ret = datastore.select(range(shard[0], shard[1])).map(function=function, with_indices=with_indices, input_columns=input_columns,
                      batched=batched, batch_size=batch_size, drop_last_batch=drop_last_batch, 
                      remove_columns=remove_columns, keep_in_memory=keep_in_memory, 
                      load_from_cache_file=load_from_cache_file, cache_file_name=cache_file_name,
                      writer_batch_size=writer_batch_size, features=features,
                      disable_nullable=disable_nullable, fn_kwargs=fn_kwargs,
                      num_proc=num_proc, suffix_template=suffix_template,
                      new_fingerprint=new_fingerprint, desc=desc,distributed_context=None)
        ret.save_to_disk(dataset_path)
        output_igzip_files = glob.glob(curr_task_subfile_path+".*")
        if sort:
          for f in output_igzip_files:
            Datastore.sort_file(f, cache_dir=cache_dir, gzip_output=gzip_output)
        if [_ for r in curr_result_subfile_path if r]:
          for input_file_path, _ in input_files:
            if (delete_input_files_on_finalize or input_file_path.startswith("__result")):
                os.unlink(input_file_path)
                if input_file_path.endswith(".gz") and os.path.exists(input_file_path.repalce(".gz", ".igz")):
                  os.unlink(input_file_path.repalce(".gz", ".igz"))

        ret= Datastore.merge_and_save_files(output_igzip_files, curr_task_subfile_path, sort, shared_dir, gzip_output=gzip_output)
        # add in the other info for this shard and return the complete shard with ranges in json format
        return ret

    def init_map_reduce(self, *args, **kwargs):
      self.map_reduce_args=[args,kwargs]
      if kwargs.get('input_file_function'):
        self.input_files = kwargs.get('input_file_function')(self)
      return self

    @staticmethod
    def upsert_sql_from_batch(batch, views_map, id_feature, src_feature_to_dst_views_map):
      sql_results={}
      for src_feature, dst_feature in src_feature_to_dst_views_map.items() if src_feature_to_dst_views_map is not None else zip(batch.keys(),batch.keys()):
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
    #Updating views might create an unepxected side-effect on caching.  Use caching with cuation when editing views.
    def map(self, 
        function: Optional[Callable] = None,
        with_indices: bool = False,
        input_columns: Optional[Union[str, List[str]]] = None,
        batched: bool = True,
        batch_size: Optional[int] = 1000,
        drop_last_batch: bool = False,
        remove_columns: Optional[List[str]] = None,
        keep_columns: Optional[List[str]] = None,
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
        desc: Optional[str] = None,
        handle_views: int = STATIC_VIEWS,
        output_igzip_file: Optional[str] =None,
        output_igzip_file_schema: List = None,
        keep_features: List = None, 
        cache_dir: Optional[str] =None,  
        distributed_context: DistributedContext = None, # we only use the next parameters if there is a distributed_context.
        intermediate_sort: Optional[bool] = True, 
        final_reduce: Optional[bool] =True,  
        shared_dir: Optional[str] =None, 
        gzip_output: Optional[bool]=True,
        delete_input_files_on_finalize: Optional[bool] = True,
        #add_memmap_views=None,
        #add_sql_views=None,
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
      if shared_dir is None:
        shared_dir = self.shared_dir
      if distributed_context is None:
        distributed_context = self.distributed_context
      
      #let's see if the data is broken by shards. if not, then we are doing regular map without distributed context. 
      #and we need o synch to the shared drive
      shards = []
      for key, val in self.views_map:
        if (val['type'] == 'igzip' and type(val['path']) is list):
          for input_file, start, end in val['path']:
              shards.append((start, end))

      if not shared_dir or not distributed_context or not shards:
        ret= self.map(function=function, with_indices=with_indices, input_columns=input_columns,
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
        
      else:
        self.save_to_disk(Path(shared_dir, self.output_dir))
        kwds_per_shard = [dict(Path(shared_dir, self.output_dir,),
                      shard, function=function, with_indices=with_indices, input_columns=input_columns,
                      batched=batched, batch_size=batch_size, drop_last_batch=drop_last_batch, 
                      remove_columns=remove_columns, keep_in_memory=keep_in_memory, 
                      load_from_cache_file=load_from_cache_file, cache_file_name=cache_file_name,
                      writer_batch_size=writer_batch_size, features=features,
                      disable_nullable=disable_nullable, fn_kwargs=fn_kwargs,
                      num_proc=num_proc, suffix_template=suffix_template,
                      new_fingerprint=new_fingerprint, desc=desc,
                      handle_views=handle_views,
                      )
                      for shard in shards
                  ]

        shard_file_and_ranges = [r.result() for r in self.distributed_context.map(Datastore._distributed_map, kwds_per_shard)]
        shard_file_and_ranges = [r for r in shard_file_and_ranges if r]
        if final_sort_reduce: # there is a case where the final reduce is just a concat?
          #, split_lines=5000000, lock=True
            shard_file_and_ranges = self.merge_and_save_files(shard_file_and_ranges, output_igzip_file, intermediate_sort, shared_dir, gzip_output)
        # now see if the schema includes any other views
        shutil.mkdir(Path(shared_dir, output_igzip_file))
        feature_views = {}
        if type(output_igzip_file_schema) is dict:
          output_igzip_file_schema = list(output_igzip_file_schema.items())
        for column, feature_dtype in enumerate(output_igzip_file_schema):
          feature, dtype = feature_dtype
          feature_views[feature] = {'type': 'igzip', 'col': column, 'dtype': dtype, 'file_type': shard_file_and_ranges[0][0].split(".")[-2], 'path': shard_file_and_ranges}
        if keep_columns:
          keep_columns = list(set(keep_columns+[self.id_feature]))
          for view in keep_columns:
            if view in self.feature_views:
              feature_views[view] = copy.deepcopy(self.feature_views[view])
          for column in self.columns:
            if column not in keep_columns:
                self = self.remove_columns(column)
          if shard_file_ranges[-1][-1] < len(self):
            self = self.select(range(shard_file_ranges[-1][-1]))
          ret = Datastore.from_dataset(self, self, feature_views=feature_views, output_dir=output_dir)
        else:
          ret = Datastore.from_dataset(Datastore.from_dict({self.id_feature: range(shard_file_ranges[-1][-1])}), self, feature_views=feature_views, output_dir=output_dir)

        for column in remove_columns if remove_columns is not None else []:
          if column in self.views_map and column in ret:
            print (f"warning: the map function returned a column {column} which is the same as a detached view. this column will be persisted to arrow.")
        ret.save_to_disk(Path(shared_dir, output_igzip_file), move_files=True, shared_dir=shared_dir, cache_dir=cache_dir)
        if clear_cache:
            dataset_path = os.path.dirname(self.cache_files[0]['filename'])
            if os.path.isdir(dataset_path):
                logger.warning(f"Clearing cache at {dataset_path}")
                shutil.rmtree(builder._cache_dir)
            download_dir = os.path.join(self.cache_dir, datasets.config.DOWNLOADED_DATASETS_DIR)
            if os.path.isdir(download_dir):
                logger.warning(f"Clearing cache at {download_dir}")
                shutil.rmtree(download_dir)

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
            if 'connection_url' in value:
              if "sqlite:///" in value['connection_url']:
                src = value['connection_url'].replace("sqlite:///", "")
                if value['connection_url'] in Datastore.db_connection:
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

            # if the src is not the same as dest, we want to move or copy
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
                "id2idx_identity",
                "id_feature",
                "pipelines_manager"
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


### Some testing routines in particular to test on OSCAR

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

def _download_urls(urls):
  for url in urls:
    if not os.path.exists(url.split("/")[-1]):
      os.system(f"wget {url}")
    data = Datastore.from_dict({}).add_igzip("text", url.split("/")[-1])
    print (data[-1])


if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    from datasets import load_dataset
    import fsspec, requests, aiohttp
    # some test code to manage oscar downloading
    args = sys.argv[1:]
    if not args:
      exit()
    if "-launch_dask_node" == args[0]:
      scheduler_file = args[1]
      DistributedContext.start(scheduler_file=scheduler_file)
        
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

    if "-test_sqlite_fs" in args:
      db = DatabaseExt("sqlite:///test.db")
      books_table = db['books']
      books_table.create_fts_index_column('text', stemmer="unicode61")
      for t in """In Which We Are Introduced to Winnie the Pooh and Some Bees and the Stories Begin
Winnie-the-Pooh is out of honey, so he and Christopher Robin attempt to trick some bees out of theirs, with disastrous results.
In Which Pooh Goes Visiting and Gets into a Tight Place
Pooh visits Rabbit, but eats so much while in Rabbit's house that he gets stuck in Rabbit's door on the way out.
In Which Pooh and Piglet Go Hunting and Nearly Catch a Woozle
Pooh and Piglet track increasing numbers of footsteps round and round a spinney of trees.
In Which Eeyore Loses a Tail and Pooh Finds One
Pooh sets out to find Eeyore's missing tail, and notices something interesting about Owl's bell-pull.
In Which Piglet Meets a Heffalump
Piglet and Pooh try to trap a Heffalump, but wind up trapping the wrong sort of creature.
In Which Eeyore has a Birthday and Gets Two Presents
Pooh feels bad that no one has gotten Eeyore anything for his birthday, so he and Piglet try their best to get him presents.
In Which Kanga and Baby Roo Come to the Forest and Piglet has a Bath
Rabbit convinces Pooh and Piglet to try to kidnap newcomer Baby Roo to convince newcomer Kanga to leave the forest.
In Which Christopher Robin Leads an Expotition to the North Pole
Christopher Robin and all of the animals in the forest go on a quest to find the North Pole in the Hundred Acre Wood.
In Which Piglet is Entirely Surrounded by Water
Piglet is trapped in his home by a flood, so he sends a message out in a bottle in hope of rescue.
In Which Christopher Robin Gives Pooh a Party and We Say Goodbye
Christopher Robin gives Pooh a party for helping to rescue Piglet during the flood.
The central character in the series is Harry Potter, a boy who lives in the fictional town of Little Whinging, Surrey with his aunt, uncle, and cousin – the Dursleys – and discovers at the age of eleven that he is a wizard, though he lives in the ordinary world of non-magical people known as Muggles.[8] The wizarding world exists parallel to the Muggle world, albeit hidden and in secrecy. His magical ability is inborn, and children with such abilities are invited to attend exclusive magic schools that teach the necessary skills to succeed in the wizarding world.[9]

Harry becomes a student at Hogwarts School of Witchcraft and Wizardry, a wizarding academy in Scotland, and it is here where most of the events in the series take place. As Harry develops through his adolescence, he learns to overcome the problems that face him: magical, social, and emotional, including ordinary teenage challenges such as friendships, infatuation, romantic relationships, schoolwork and exams, anxiety, depression, stress, and the greater test of preparing himself for the confrontation that lies ahead in wizarding Britain's increasingly-violent second wizarding war.[10]

Each novel chronicles one year in Harry's life[11] during the period from 1991 to 1998.[12] The books also contain many flashbacks, which are frequently experienced by Harry viewing the memories of other characters in a device called a Pensieve.

The environment Rowling created is intimately connected to reality. The British magical community of the Harry Potter books is inspired by 1990s British culture, European folklore, classical mythology and alchemy, incorporating objects and wildlife such as magic wands, magic plants, potions, spells, flying broomsticks, centaurs and other magical creatures, and the Philosopher's Stone, beside others invented by Rowling. While the fantasy land of Narnia is an alternate universe and the Lord of the Rings' Middle-earth a mythic past, the wizarding world of Harry Potter exists parallel to the real world and contains magical versions of the ordinary elements of everyday life, with the action mostly set in Scotland (Hogwarts), the West Country, Devon, London, and Surrey in southeast England.[13] The world only accessible to wizards and magical beings comprises a fragmented collection of overlooked hidden streets, ancient pubs, lonely country manors, and secluded castles invisible to the Muggle population.[9]

Early years
When the first novel of the series, Harry Potter and the Philosopher's Stone, opens, it is apparent that some significant event has taken place in the wizarding world – an event so very remarkable that even Muggles (non-magical people) notice signs of it. The full background to this event and Harry Potter's past is revealed gradually throughout the series. After the introductory chapter, the book leaps forward to a time shortly before Harry Potter's eleventh birthday, and it is at this point that his magical background begins to be revealed.

Despite Harry's aunt and uncle's desperate prevention of Harry learning about his abilities,[14] their efforts are in vain. Harry meets a half-giant, Rubeus Hagrid, who is also his first contact with the wizarding world. Hagrid reveals himself to be the Keeper of Keys and Grounds at Hogwarts as well as some of Harry's history.[14] Harry learns that, as a baby, he witnessed his parents' murder by the power-obsessed dark wizard Lord Voldemort (more commonly known by the magical community as You-Know-Who or He-Who-Must-Not-Be-Named, and by Albus Dumbledore as Tom Marvolo Riddle) who subsequently attempted to kill him as well.[14] Instead, the unexpected happened: Harry survived with only a lightning-shaped scar on his forehead as a memento of the attack, and Voldemort disappeared soon afterwards, gravely weakened by his own rebounding curse.

As its inadvertent saviour from Voldemort's reign of terror, Harry has become a living legend in the wizarding world. However, at the orders of the venerable and well-known wizard Albus Dumbledore, the orphaned Harry had been placed in the home of his unpleasant Muggle relatives, the Dursleys, who have kept him safe but treated him poorly, including confining him to a cupboard without meals and treating him as their servant. Hagrid then officially invites Harry to attend Hogwarts School of Witchcraft and Wizardry, a famous magic school in Scotland that educates young teenagers on their magical development for seven years, from age eleven to seventeen.

With Hagrid's help, Harry prepares for and undertakes his first year of study at Hogwarts. As Harry begins to explore the magical world, the reader is introduced to many of the primary locations used throughout the series. Harry meets most of the main characters and gains his two closest friends: Ron Weasley, a fun-loving member of an ancient, large, happy, but poor wizarding family, and Hermione Granger, a gifted, bright, and hardworking witch of non-magical parentage.[14][15] Harry also encounters the school's potions master, Severus Snape, who displays a conspicuously deep and abiding dislike for him, the rich brat Draco Malfoy whom he quickly makes enemies with, and the Defence Against the Dark Arts teacher, Quirinus Quirrell, who later turns out to be allied with Lord Voldemort. He also discovers a talent of flying on broomsticks and is recruited for his house's Quidditch team, a sport in the wizarding world where players fly on broomsticks. The first book concludes with Harry's second confrontation with Lord Voldemort, who, in his quest to regain a body, yearns to gain the power of the Philosopher's Stone, a substance that bestows everlasting life and turns any metal into pure gold.[14]

The series continues with Harry Potter and the Chamber of Secrets, describing Harry's second year at Hogwarts. He and his friends investigate a 50-year-old mystery that appears uncannily related to recent sinister events at the school. Ron's younger sister, Ginny Weasley, enrols in her first year at Hogwarts, and finds an old notebook in her belongings which turns out to be the diary of a previous student, Tom Marvolo Riddle, written during World War II. He is later revealed to be Voldemort's younger self, who is bent on ridding the school of "mudbloods", a derogatory term describing wizards and witches of non-magical parentage. The memory of Tom Riddle resides inside of the diary and when Ginny begins to confide in the diary, Voldemort is able to possess her.

Through the diary, Ginny acts on Voldemort's orders and unconsciously opens the "Chamber of Secrets", unleashing an ancient monster, later revealed to be a basilisk, which begins attacking students at Hogwarts. It kills those who make direct eye contact with it and petrifies those who look at it indirectly. The book also introduces a new Defence Against the Dark Arts teacher, Gilderoy Lockhart, a highly cheerful, self-conceited wizard with a pretentious facade, later turning out to be a fraud. Harry discovers that prejudice exists in the Wizarding World through delving into the school's history, and learns that Voldemort's reign of terror was often directed at wizards and witches who were descended from Muggles.

Harry also learns that his ability to speak the snake language Parseltongue is rare and often associated with the Dark Arts. When Hermione is attacked and petrified, Harry and Ron finally piece together the puzzles and unlock the Chamber of Secrets, with Harry destroying the diary for good and saving Ginny, and, as they learn later, also destroying a part of Voldemort's soul. The end of the book reveals Lucius Malfoy, Draco's father and rival of Ron and Ginny's father, to be the culprit who slipped the book into Ginny's belongings.

The third novel, Harry Potter and the Prisoner of Azkaban, follows Harry in his third year of magical education. It is the only book in the series which does not feature Lord Voldemort in any form, only being mentioned. Instead, Harry must deal with the knowledge that he has been targeted by Sirius Black, his father's best friend, and, according to the Wizarding World, an escaped mass murderer who assisted in the murder of Harry's parents. As Harry struggles with his reaction to the dementors – dark creatures with the power to devour a human soul and feed on despair – which are ostensibly protecting the school, he reaches out to Remus Lupin, a Defence Against the Dark Arts teacher who is eventually revealed to be a werewolf. Lupin teaches Harry defensive measures which are well above the level of magic generally executed by people his age. Harry comes to know that both Lupin and Black were best friends of his father and that Black was framed by their fourth friend, Peter Pettigrew, who had been hiding as Ron's pet rat, Scabbers.[16] In this book, a recurring theme throughout the series is emphasised – in every book there is a new Defence Against the Dark Arts teacher, none of whom lasts more than one school year.

Voldemort returns
"The Elephant House", a small, painted red café where Rowling wrote a few chapters of Harry Potter and the Philosopher's Stone
The Elephant House was one of the cafés in Edinburgh where Rowling wrote the first part of Harry Potter.

The former 1st floor Nicholson's Cafe now renamed Spoon in Edinburgh where J. K. Rowling wrote the first few chapters of Harry Potter and the Philosopher’s Stone.

The J. K. Rowling plaque on the corner of the former Nicholson's Cafe (now renamed Spoon) at 6A Nicolson St, Edinburgh.
During Harry's fourth year of school (detailed in Harry Potter and the Goblet of Fire), Harry is unwillingly entered as a participant in the Triwizard Tournament, a dangerous yet exciting contest where three "champions", one from each participating school, must compete with each other in three tasks in order to win the Triwizard Cup. This year, Harry must compete against a witch and a wizard "champion" from overseas schools Beauxbatons and Durmstrang, as well as another Hogwarts student, causing Harry's friends to distance themselves from him.[17]

Harry is guided through the tournament by their new Defence Against the Dark Arts professor, Alastor "Mad-Eye" Moody, who turns out to be an impostor – one of Voldemort's supporters named Barty Crouch, Jr. in disguise, who secretly entered Harry's name into the tournament. The point at which the mystery is unravelled marks the series' shift from foreboding and uncertainty into open conflict. Voldemort's plan to have Crouch use the tournament to bring Harry to Voldemort succeeds. Although Harry manages to escape, Cedric Diggory, the other Hogwarts champion in the tournament, is killed by Peter Pettigrew and Voldemort re-enters the Wizarding World with a physical body.

In the fifth book, Harry Potter and the Order of the Phoenix, Harry must confront the newly resurfaced Voldemort. In response to Voldemort's reappearance, Dumbledore re-activates the Order of the Phoenix, a secret society which works from Sirius Black's dark family home to defeat Voldemort's minions and protect Voldemort's targets, especially Harry. Despite Harry's description of Voldemort's recent activities, the Ministry of Magic and many others in the magical world refuse to believe that Voldemort has returned. In an attempt to counter and eventually discredit Dumbledore, who along with Harry is the most prominent voice in the Wizarding World attempting to warn of Voldemort's return, the Ministry appoints Dolores Umbridge as the High Inquisitor of Hogwarts and the new Defence Against the Dark Arts teacher. She transforms the school into a dictatorial regime and refuses to allow the students to learn ways to defend themselves against dark magic.[18]

Hermione and Ron form "Dumbledore's Army", a secret study group in which Harry agrees to teach his classmates the higher-level skills of Defence Against the Dark Arts that he has learned from his previous encounters with Dark wizards. Through those lessons, Harry begins to develop a crush on the popular and attractive Cho Chang. Juggling schoolwork, Umbridge's incessant and persistent efforts to land him in trouble and the defensive lessons, Harry begins to lose sleep as he constantly receives disturbing dreams about a dark corridor in the Ministry of Magic, followed by a burning desire to learn more. An important prophecy concerning Harry and Lord Voldemort is then revealed,[19] and Harry discovers that he and Voldemort have a painful connection, allowing Harry to view some of Voldemort's actions telepathically. In the novel's climax, Harry is tricked into seeing Sirius tortured and races to the Ministry of Magic. He and his friends face off against Voldemort's followers (nicknamed Death Eaters) at the Ministry of Magic. Although the timely arrival of members of the Order of the Phoenix saves the teenagers' lives, Sirius Black is killed in the conflict.

In the sixth book, Harry Potter and the Half-Blood Prince, Voldemort begins waging open warfare. Harry and his friends are relatively protected from that danger at Hogwarts. They are subject to all the difficulties of adolescence – Harry eventually begins dating Ginny, Ron establishes a strong infatuation with fellow Hogwarts student Lavender Brown, and Hermione starts to develop romantic feelings towards Ron. Near the beginning of the novel, lacking his own book, Harry is given an old potions textbook filled with many annotations and recommendations signed by a mysterious writer titled; "the Half-Blood Prince". This book is a source of scholastic success and great recognition from their new potions master, Horace Slughorn, but because of the potency of the spells that are written in it, becomes a source of concern.

With war drawing near, Harry takes private lessons with Dumbledore, who shows him various memories concerning the early life of Voldemort in a device called a Pensieve. These reveal that in order to preserve his life, Voldemort has split his soul into pieces, used to create a series of Horcruxes – evil enchanted items hidden in various locations, one of which was the diary destroyed in the second book.[20] Draco, who has joined with the Death Eaters, attempts to attack Dumbledore upon his return from collecting a Horcrux, and the book culminates in the killing of Dumbledore by Professor Snape, the titular Half-Blood Prince.

Harry Potter and the Deathly Hallows, the last original novel in the series, begins directly after the events of the sixth book. Lord Voldemort has completed his ascension to power and gained control of the Ministry of Magic. Harry, Ron and Hermione drop out of school so that they can find and destroy Voldemort's remaining Horcruxes. To ensure their own safety as well as that of their family and friends, they are forced to isolate themselves. A ghoul pretends to be Ron ill with a contagious disease, Harry and the Dursleys separate, and Hermione wipes her parents' memories and sends them abroad.

As the trio searches for the Horcruxes, they learn details about an ancient prophecy of the Deathly Hallows, three legendary items that when united under one Keeper, would supposedly allow that person to be the Master of Death. Harry discovers his handy Invisibility Cloak to be one of those items, and Voldemort to be searching for another: the Elder Wand, the most powerful wand in history. At the end of the book, Harry and his friends learn about Dumbledore's past, as well as Snape's true motives – he had worked on Dumbledore's behalf since the murder of Harry's mother. Eventually, Snape is killed by Voldemort out of paranoia.

The book culminates in the Battle of Hogwarts. Harry, Ron and Hermione, in conjunction with members of the Order of the Phoenix and many of the teachers and students, defend Hogwarts from Voldemort, his Death Eaters, and various dangerous magical creatures. Several major characters are killed in the first wave of the battle, including Remus Lupin and Fred Weasley, Ron's older brother. After learning that he himself is a Horcrux, Harry surrenders himself to Voldemort in the Forbidden Forest, who casts a killing curse (Avada Kedavra) at him. The defenders of Hogwarts do not surrender after learning of Harry's presumed death and continue to fight on. Harry awakens and faces Voldemort, whose Horcruxes have all been destroyed. In the final battle, Voldemort's killing curse rebounds off Harry's defensive spell (Expelliarmus), killing Voldemort.

An epilogue "Nineteen Years Later"[21] describes the lives of the surviving characters and the effects of Voldemort's death on the Wizarding World. In the epilogue, Harry and Ginny are married with three children, and Ron and Hermione are married with two children.[22]
""".split("\n"):
        if t.strip():
            books_table.insert({'text': t})
      print(list(books_table.find(id={'in': (3,4)})))
      print(list(books_table.find()))
      print("Bottle*", list(books_table.find(_fts_query=[('text','Bottle*')])))
      print("robin", list(books_table.find(_fts_query=[('text','robin')])))
      print("Home", list(books_table.find(_fts_query=[('text','Home')])))
      print("notic*", list(books_table.find(_fts_query=[('text','notic*')])))
      print("sign* AND notic*", list(books_table.find(_fts_query=[('text','sign* AND notic*')])))
      print("sign AND notic*", list(books_table.find(_fts_query=[('text','sign AND notic*')])))
      

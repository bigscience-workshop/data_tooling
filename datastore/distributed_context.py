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

  """Distributed context over Dask, torch.distributed and MPI to perform distributed processing, using a shared file for finding the scheduler, with an optional streamlit app attached"""

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

    postgress_connection_uri="postgresql://{}:{}@{}?port={}&dbname={}".format(
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


#%%writefile data_tooling/distributed.py
#Copyright July 2021 Ontocord LLC. Licensed under Apache v2 https://www.apache.org/licenses/LICENSE-2.0
# based on https://github.com/gstaff/flask-ngrok which is licensed under Apache v2
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
try:
  from data_tooling.datastore_utils import *
except:
  from datastore_utils import *

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


#TODO: streamlit does tracking. For privacy and security concern, we need to turn this off.
# this class provides Dask and Streamlit through ngrok.
# for example:
# DistributedContext.start(start_dask=True, shared_scheduler_file="<my scheduler file>", start_streamlit=True, streamlit_app_file="<my app file>.py")
# st = DistributedContext()
# if you do not provide a custom <my app file>.py file, an app file will automatically be created, and you can write logs to the app like so:
#
# st.write("test")
# or
# with st:
#   print ("test")

class DistributedContext:

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
                f.write(f"st.{command}(\"\"\""+str(an_item)+"\"\"\")\n")
        else:
              f.write(f"st.{command}(\"\"\""+str(item)+"\"\"\")\n")
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

  ngrok = None
  dask_node_id = None
  dask_nodes = {}
  dask_client = None
  dask_scheduler_file = None
  streamlit_app_file = None
  streamlit_process = None

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

if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    args = sys.argv[1:]
    if "-launch_dask_node" == args[0]:
      scheduler_file = args[1]
      DistributedContext.launch_dask_node(scheduler_file)

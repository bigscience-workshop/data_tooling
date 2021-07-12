#%%writefile data-tooling/colab_ngrok_dask.py
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

from filelock import UnixFileLock 
# FileLock does not really work for gdrive.  

class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# Not thread or multiprocessing safe
class ColabGDriveFileLock(UnixFileLock):
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

def tcp_run_ngrok(port, hostname=None, server_mode=False):
        command = _get_command()
        ngrok_path = str(Path(tempfile.gettempdir(), "ngrok"))
        _download_ngrok(ngrok_path)
        executable = str(Path(ngrok_path, command))
        os.chmod(executable, 0o777)
        if hostname is not None:
          if server_mode:
            ngrok = subprocess.Popen([executable, 'tcp', '-region=us', '-hostname='+hostname, str(port)])
          else:
            ngrok = subprocess.Popen([executable, 'tcp', '-region=us', '-hostname='+hostname, str(port)], preexec_fn = preexec)
        else:
          if server_mode:
            ngrok = subprocess.Popen([executable, 'tcp', '-region=us',  str(port)]) 
          else:
            ngrok = subprocess.Popen([executable, 'tcp', '-region=us',  str(port)], preexec_fn = preexec) 
        atexit.register(ngrok.terminate)
        localhost_url = "http://localhost:4040/api/tunnels"  # Url with tunnel details
        for i in range(10):
          time.sleep(5)
          try:
            tunnel_url = requests.get(localhost_url).text  # Get the tunnel information
            j = json.loads(tunnel_url)
            tunnel_url = j['tunnels'][0]['public_url']  # Do the parsing of the get
            tunnel_url = tunnel_url.replace("https", "http")
            return tunnel_url
          except:
            pass

        return 'error'

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
        download_path = _download_file(url)
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(ngrok_path)

def _download_file(url):
        local_filename = url.split('/')[-1]
        r = requests.get(url, stream=True)
        download_path = str(Path(tempfile.gettempdir(), local_filename))
        with open(download_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        return download_path
    

def write_authtoken(token):
      if not os.path.exists("/root/.ngrok2"):
        os.mkdir("/root/.ngrok2")
      with open("/root/.ngrok2/ngrok.yml", "w") as f:
        f.write("authtoken: "+token+"\n")

# you will need an ngrok account that has tcp capabilities. 
def run_dask_ngrok(port, hostname=None, server_mode=False):
  token = getpass('ngrok token: ') 
  token = urllib.parse.quote(token)
  write_authtoken(token)
  ngrok_address = tcp_run_ngrok(port, hostname, server_mode)
  #print (ngrok_address)
  return ngrok_address

dask_node_id = None
dask_nodes = {}
dask_client = None
def launch_dask(shared_scheduler_file, num_procs=4, time_out=43200): # timeout in 12 hours
  global dask_node_id, dask_nodes
  #If there is no master scheduler node, this node will become the master scheduler node
  with ColabGDriveFileLock(shared_scheduler_file+".lock"):
    if os.path.exists(shared_scheduler_file):
      next(wait_until_files_loaded(shared_scheduler_file))
      with open(shared_scheduler_file, "r") as f:
        for line in f.read().split("\n"):
          address, node_id, last_heartbeat = line.split("\t")
          last_heartbeat = datetime.fromtimestamp(last_heartbeat)
          if datetime.timestamp(datetime.now()) - datetime.fromtimestamp(last_heartbeat) > time_out: 
            print (f"node {node_id} has timed out")
          else:
            dask_nodes[node_id] = (address, last_heartbeat)
    if 0 not in dask_nodes:
        dask_node_id = 0
        dask_client = Client(n_workers=num_procs, name=f"worker_{dask_node_id}")
        port = int(str(dask_client).split("tcp://")[1].strip().split()[0].strip("' ").split(":")[-1].strip())
        dask_scheduler = run_dask_ngrok(port)
        dask_nodes[dask_node_id] = (dask_scheduler, datetime.timestamp(datetime.now()))
        with open(shared_scheduler_file, "w") as f: 
          for key, val in dask_nodes.items():
            f.write(str(key)+"\t"+str(val[0])+"\t"+str(val[1])+"\n")
        return dask_client
    else:
        address = dask_nodes[0][0]
        dask_node_id = len(dask_nodes)
        # todo, pipe stderr and check if there are errors.
        dask = subprocess.Popen(["dask-worker", address, '--name', f"worker_{dask_node_id}", "--nprocs", num_proces, "--nthreads", "1", "--no-dashboard"], preexec_fn = preexec) 
        atexit.register(dask.terminate)
        #!dask-worker $address --name $dask_node_id --nprocs $num_procs --nthreads 1  --no-dashboard 
        # if there is an error in connecting to the scheduler, then the scheduler has probably died and we need to create a new scheduler with this node.
        dask_nodes[dask_node_id] = (dask_node_id, datetime.timestamp(datetime.now()))
        with open(shared_scheduler_file, "w") as f: 
          for key, val in dask_nodes.items():
            f.write(str(key)+"\t"+str(val[0])+"\t"+str(val[1])+"\n")
        return None

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

""" Common utilities for datastorage"""


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
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir, os.path.pardir)))


logger = logging.get_logger(__name__)

def preexec(): # Don't forward signals.
    os.setpgrp()

	
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
  if isinstance(flist, str):
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

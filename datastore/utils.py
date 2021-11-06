# Copyright 2021, Ontocord, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Common utilities for datastorage"""
import os
import sys

from datasets.utils import logging
from filelock import FileLock, UnixFileLock

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir
        )
    )
)

logger = logging.get_logger(__name__)
####################################################################################################
# some utils

# if an array is contiguous, return True, and the start and end+1 range usable in 'range(start, end)'
def is_contiguous(arr):
    start = None
    prev = None
    contiguous = True
    for i in arr:
        if start is None:
            start = i
        if prev is None or i == prev + 1:
            prev = i
            continue
        contiguous = False
        break
    return contiguous, start, i + 1


# This is used for seeing if files from a remote drive have finished loading. Files could be in flight while we try to retreive them.
def wait_until_files_loaded(flist, max_tries=120, fs=None):  # wait 2 hrs max
    if fs is None:
        fs = os
    if isinstance(flist, str):
        flist = [[flist, 0]]
    else:
        flist = [[f, 0] for f in flist]
    for j in range(len(flist) * max_tries):
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
                flist[i] = [f, incr]
        if num_done == len(flist):
            return
    return


class DummyLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# FileLock does not really work for gdrive or other types of shared
# drives (ftp).  So we create a SharedFileLock instead, which is not
# guranteed to lock a file, but the best we have.  This is because if
# two nodes are trying to write to the same file, gdrive will for
# example create a file.txt and a file(1).txt as the other file being
# written to.
class SharedFileLock(UnixFileLock):
    def __init__(self, lock_file, timeout=-1, locks=None):
        super().__init__(lock_file, timeout)
        self.locks = locks
        self.locked = False

    def __enter__(self):
        if self.locks is not None and self._lock_file not in self.locks:
            self.locks[self._lock_file] = 1
            self.acquire()
            self.locked = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.locked:
            self.release()
            if self.locks is not None and self._lock_file in self.locks:
                del self.locks[self._lock_file]
            self.locked = False
        return None

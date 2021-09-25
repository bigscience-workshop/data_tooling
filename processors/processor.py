# coding=utf-8
# Copyright, 2021 Ontocord, LLC, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from time import time
import numpy as np
from collections import Counter
from itertools import chain
import os
import re
import glob
import math
from nltk.corpus import stopwords
import difflib
import random
import nltk
from random import choice
import spacy, neuralcoref, itertools
from collections import Counter, OrderedDict
trannum = str.maketrans("0123456789", "1111111111")

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir, os.path.pardir)))



# TODO, add data lineage and processing meta-data, inlcuding processor state varaibles like random seeds. store as meta-data.
# text examples can be processed and merged, and split. 
# a data-point is composed of a (row_id, doc_id). 
# data-linage is stored as a tree, with the root node as the current data point.
# the goal is we want to be able to identify a particular text example for removal or recompute based on 
# contestation.
# we want to also be able to recreate examples from the metadata. 

class ProcessorPipeline:
  """
  Used to pre-process text. Processes text in a sequence of processors. 
  """

  def __init__(self, processors):
    self.processors = processors
  
  def process(self, text="", batch=None, *args, **argv):
    ret = []
    if batch is None:
      batch = [text]
    for processor in self.processors:
      ret =  processor.process(batch=ret, *args, **argv) 
    return ret


class Processor:
  def process(self, batch,  text="", batch=None, *args, **argv):
    raise NotImpelmentedError()

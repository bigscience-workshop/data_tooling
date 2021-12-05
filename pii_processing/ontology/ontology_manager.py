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
from random import sample
import glob, os, re
import multiprocessing

import gzip
import  os, argparse
import itertools
from collections import Counter, OrderedDict
import os
import json
import threading
import numpy as np
import os
import time
import json
import copy

from time import time
import numpy as np
from collections import Counter
from itertools import chain
import glob
import json
import math, os
import random
import transformers
import sys, os
import json
import faker
import gzip
from faker.providers import person, job

from collections import Counter
import re
import gzip
import urllib
import re
from transformers import AutoTokenizer
from nltk.corpus import stopwords

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir, os.path.pardir)))
from data_tooling.pii_processing.ontology.stopwords import stopwords as stopwords_ac_dc
mt5_underscore= "▁"
trannum = str.maketrans("0123456789", "1111111111")

class OntologyManager:
  """ 
  Basic ontology manager. Stores the upper ontology and lexicon that
  maps to the leaves of the ontology.  Has functions to determine
  whether a word is in the ontology, and to tokenize a sentence with
  words from the ontology.
  """

  default_strip_chars="-,~`.?!@#$%^&*(){}[]|\\/-_+=<>;'\""
  stopwords_wn = set(stopwords.words())
  x_lingual_onto_name = "yago_cn_wn"
  default_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

  default_label2label = {'SOC_ECO_CLASS':'NORP', 
      'RACE':'NORP', 
      'POLITICAL_PARTY':'NORP', 
      'UNION':'NORP', 
      'RELIGION':'NORP', 
      'RELIGION_MEMBER': 'NORP',
      'POLITICAL_PARTY_MEMBER': 'NORP',
      'UNION_MEMBER': 'NORP'
      }

  default_upper_ontology =  { 
      'PERSON': ['PERSON'],
      'PUBLIC_FIGURE': ['PUBLIC_FIGURE', 'PERSON'],      
      'ORG': ['ORG'],
      'NORP': ['NORP', 'ORG'],
      'AGE': ['AGE'],
      'DISEASE': ['DISEASE'],      
      'STREET_ADDRESS': ['STREET_ADDRESS', 'LOC'],
      'GPE': ['GPE', 'LOC'],
      'CREDIT_CARD': ['CREDIT_CARD', 'CARDINAL'],
      'EMAIL_ADDRESS': ['EMAIL_ADDRESS', 'ELECTRONIC_ADDRESS'],
      'GOVT_ID': ['GOVT_ID', 'CARDINAL'],
  }  
  def __init__(self, target_lang="", data_dir=None,  tmp_dir=None, max_word_len=4, compound_word_step =3,  strip_chars=None,  \
                 upper_ontology=None,  x_lingual_lexicon_by_prefix_file="lexicon_by_prefix.json.gz", target_lang_data_file=None, x_lingual2ner_file=None, \
                 connector = "_", label2label=None, min_word_len=5):
    self.mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    self.target_lang_lexicon = {}
    self.x_lingual_lexicon_by_prefix = {}
    self.target_lang = target_lang
    self.stopwords = set(stopwords_ac_dc.get(target_lang,[])+ list(self.stopwords_wn))
    self._max_lexicon = 0
    if data_dir is None: data_dir = self.default_data_dir 
    if tmp_dir is None: tmp_dir = "/tmp/pii_processing/"
    os.system(f"mkdir -p {data_dir}")
    os.system(f"mkdir -p {tmp_dir}")
    self.tmp_dir = tmp_dir
    self.data_dir = data_dir
    if strip_chars is None:
      strip_chars = self.default_strip_chars
    self.strip_chars_set = set(strip_chars)
    self.strip_chars = strip_chars
    self.connector = connector
    self.max_word_len = max_word_len
    self.min_word_len = min_word_len
    self.compound_word_step = compound_word_step
    if label2label is None:
      label2label = self.default_label2label
    self.label2label=label2label
    if upper_ontology is None:
      upper_ontology = self.default_upper_ontology
    self.ontology = OrderedDict()
    self.load_upper_ontology(upper_ontology)
    self.load_x_lingual_lexicon_from_prefix_file(x_lingual_lexicon_by_prefix_file)
    if x_lingual2ner_file is not None:  
      self.load_x_lingual_lexicon_from_x_lingual2ner_file(x_lingual2ner_file)
    if target_lang_data_file is None and target_lang:
      target_lang_data_file = f"{data_dir}/{target_lang}.json"
    if target_lang_data_file is not None:
      self.load_target_lang_data(target_lang_data_file, target_lang=target_lang)
    #used for cjk processing
    
  def load_upper_ontology(self, upper_ontology):
    # TODO: load and save from json file
    if upper_ontology is None: upper_ontology =  {}

    self.upper_ontology = {}

    for key, val in upper_ontology.items():
      key = key.upper()
      if key not in self.upper_ontology:
        self.upper_ontology[key] = [val, len(self.upper_ontology)]
      else:
        self.upper_ontology[key] = [val, self.upper_ontology[key][1]]
  
  def load_x_lingual_lexicon_from_x_lingual2ner_file(self, x_lingual2ner_file):
    data_dir = self.data_dir
    tmp_dir = self.tmp_dir
    if x_lingual2ner_file is None: return
    if os.path.exists(x_lingual2ner_file):
      word2ner = json.load(open(x_lingual2ner_file, "rb"))
      self.add_to_ontology(word2ner, onto_name="yago_cn_wn")
    elif os.path.exists(os.path.join(data_dir, x_lingual2ner_file)):
      word2ner = json.load(open(os.path.join(data_dir, x_lingual2ner_file), "rb"))
      self.add_to_ontology(word2ner, onto_name=self.x_lingual_onto_name)
    else:
      print ("warning: could not find x_lingual2ner_file")

  def load_x_lingual_lexicon_from_prefix_file(self, x_lingual_lexicon_by_prefix_file="lexicon_by_prefix.json.gz"):
    data_dir = self.data_dir
    tmp_dir = self.tmp_dir
    if x_lingual_lexicon_by_prefix_file is not None:
      if not os.path.exists(x_lingual_lexicon_by_prefix_file):
        x_lingual_lexicon_by_prefix_file = f"{data_dir}/{x_lingual_lexicon_by_prefix_file}"
      if not os.path.exists(x_lingual_lexicon_by_prefix_file): 
        self.x_lingual_lexicon_by_prefix = {}
        self.ontology[self.x_lingual_onto_name] = self.x_lingual_lexicon_by_prefix
        return
      if x_lingual_lexicon_by_prefix_file.endswith(".gz"):
        with gzip.open(x_lingual_lexicon_by_prefix_file, 'r') as fin:  
          json_bytes = fin.read()                     
          json_str = json_bytes.decode('utf-8')            
          self.x_lingual_lexicon_by_prefix = json.loads(json_str)
      else:
        self.x_lingual_lexicon_by_prefix = json.load(open(x_lingual_lexicon_by_prefix_file, "rb"))
      for lexicon in self.x_lingual_lexicon_by_prefix.values():
        for val in lexicon[-1].values():
          label = val[0][0]
          if label in self.upper_ontology:
           val[0] = self.upper_ontology[label][0]
          self._max_lexicon  = max(self._max_lexicon, val[1])
    else:
      self.x_lingual_lexicon_by_prefix = {}
    self.ontology[self.x_lingual_onto_name] = self.x_lingual_lexicon_by_prefix

  def save_x_lingual_lexicon_prefix_file(self, x_lingual_lexicon_by_prefix_file="lexicon_by_prefix.json.gz"):
    """ saves the base cross lingual leixcon """
    data_dir = self.data_dir
    tmp_dir = self.tmp_dir
    #print (data_dir, x_lingual_lexicon_by_prefix_file)
    x_lingual_lexicon_by_prefix_file = x_lingual_lexicon_by_prefix_file.replace(".gz", "")
    if not x_lingual_lexicon_by_prefix_file.startswith(data_dir): 
      x_lingual_lexicon_by_prefix_file=f"{data_dir}/{x_lingual_lexicon_by_prefix_file}"  
    json.dump(self.x_lingual_lexicon_by_prefix,open(x_lingual_lexicon_by_prefix_file, "w", encoding="utf8"), indent=1)
    os.system(f"gzip {x_lingual_lexicon_by_prefix_file}")
    os.system(f"rm {x_lingual_lexicon_by_prefix_file}")

  def load_target_lang_data(self,  target_lang_data_file=None, target_lang=None):
    data_dir = self.data_dir
    tmp_dir = self.tmp_dir
    if target_lang_data_file is None:
      if os.path.exists(os.path.join(data_dir, f'{target_lang}.json')): 
        target_lang_data_file=  os.path.join(data_dir, f'{target_lang}.json')
    if target_lang_data_file is None: return
    if os.path.exists(target_lang_data_file):
      self.target_lang_data = json.load(open(target_lang_data_file, "rb"))
    else:
      self.target_lang_data = {}
    ner_regexes = []
    if 'ner_regexes' in self.target_lang_data:
      # for now we are going to ignore the PERSON rules, becaues the rules don't work yet
      # change this for Module 2 of the Hackathon.
      ner_regexes = [regex for regex in self.target_lang_data['ner_regexes'] if regex[0] != "PERSON" and regex[0] in self.upper_ontology]
      for regex in ner_regexes:
        if regex[1]:
          regex[1] = re.compile(regex[1], re.IGNORECASE)
        else:
          regex[1] = re.compile(regex[1])
    self.ner_regexes = ner_regexes

    #pronouns used for basic coref
    self.other_pronouns = set(self.target_lang_data.get('OTHER_PRONOUNS',[]))
    self.person_pronouns = set(self.target_lang_data.get('PERSON_PRONOUNS',[]))
    self.pronouns = set(list(self.other_pronouns) + list(self.person_pronouns))

    #these are used for aonymizing and de-biasing swapping. 
    #TODO: consider whether we want to create shorter/stemmed versions of these.
    self.binary_gender_swap = self.target_lang_data.get('binary_gender_swap', {})
    self.other_gender_swap = self.target_lang_data.get('other_gender_swap', {})
    self.en_pronoun2gender = self.target_lang_data.get('en_pronoun2gender', {})
    self.en_pronoun2pronoun = self.target_lang_data.get('en_pronoun2pronoun', {}) 
    self.en_pronoun2title = self.target_lang_data.get('en_pronoun2title', {})
    self.person2religion = self.target_lang_data.get('person2religion', {})  
    self.gender2en_pronoun = dict(itertools.chain(*[[(b,a) for b in lst] for a, lst in self.en_pronoun2gender.items()]))
    self.pronoun2en_pronoun = dict(itertools.chain(*[[(b,a) for b in lst] for a, lst in self.en_pronoun2pronoun.items()]))
    self.title2en_pronoun = dict(itertools.chain(*[[(b,a) for b in lst] for a, lst in self.en_pronoun2title.items()]))
    self.religion2person = dict(itertools.chain(*[[(b,a) for b in lst] for a, lst in self.person2religion.items()]))
    self.coref_window = self.target_lang_data.get('coref_window', [-1, -2, 1, 2])  #maybe this should be a parameter and not in the ontology
 
    #now specialize the ontology for target_lang and have no limit on the size of the words
    target_lang_lexicon = {}
    for label, words in self.target_lang_data if type(self.target_lang_data) is list else self.target_lang_data.items():
      if label in self.upper_ontology:
        for word in words:
          if word not in self.stopwords and (not self.cjk_detect(word) or len(word) > 1):
            target_lang_lexicon[word] = label
    #print (target_lang_lexicon)
    self.add_to_ontology(target_lang_lexicon, max_word_len=100000, onto_name=os.path.split(target_lang_data_file)[-1].split(".")[0])
    self.target_lang_lexicon = {} # target_lang_lexicon # save away the target lang ontology as a lexicon

  def save_target_lang_data(self, target_lang_data_file):
    if target_lang_data_file is None: return
    data_dir = self.data_dir
    tmp_dir = self.tmp_dir
    json.dump(self.target_lang_data,open(f"{data_dir}/{target_lang_data_file}", "w", encoding="utf8"), indent=1)
    #os.system(f"gzip {data_dir}/{target_lang_data_file}")
    
  def _has_nonstopword(self, wordArr):
    for word in wordArr:
      if word.strip(self.strip_chars) not in self.stopwords:
        return True
    return False

  def _get_all_word_shingles(self, wordArr, max_word_len=None, create_suffix_end=True):
    """  create patterned variations (prefix and suffix based shingles) """
    lenWordArr = len(wordArr)
    wordArr = [w.lower() for w in wordArr]
    if max_word_len is None: max_word_len =self.max_word_len
    compound_word_step = self.compound_word_step
    wordArr1 = wordArr2 = wordArr3 = wordArr4 = None
    ret = []
    if lenWordArr > compound_word_step:
        # we add some randomness in how we create patterns
        wordArr1 = wordArr[:compound_word_step-1] + [wordArr[-1]]
        wordArr2 = [wordArr[0]] + wordArr[1-compound_word_step:] 
        wordArr1 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr1]
        wordArr2 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr2]
        ret.extend([tuple(wordArr1),tuple(wordArr2)])
        if create_suffix_end:
          wordArr3 = copy.copy(wordArr1)
          wordArr3[-1] = wordArr3[-1] if len(wordArr3[-1]) <=max_word_len else '*'+wordArr3[-1][len(wordArr3[-1])-max_word_len+1:]
          wordArr4 = copy.copy(wordArr2)
          wordArr4[-1] = wordArr4[-1] if len(wordArr4[-1]) <=max_word_len else '*'+wordArr4[-1][len(wordArr4[-1])-max_word_len+1:]
          wordArr3 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr3]
          wordArr4 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr4]
          ret.extend([tuple(wordArr3),tuple(wordArr4)])
    else: # lenWordArr <= compound_word_step
        wordArr1 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr]
        ret.append(tuple(wordArr1))
        if lenWordArr > 1 and create_suffix_end:
          wordArr2 = copy.copy(wordArr)
          wordArr2[-1] = wordArr2[-1] if len(wordArr2[-1]) <=max_word_len else '*'+wordArr2[-1][len(wordArr2[-1])-max_word_len+1:]
          wordArr2 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr2]
          ret.append(tuple(wordArr2))
    return [list(a) for a in set(ret)]

  def add_to_ontology(self, word2ner, max_word_len=None, onto_name=None):
    """
    Add words to the ontology. The ontology is stored in compressed
    form, using an upper ontology and several subsuming prefix based lexicon
    mappings. We try to generalize the lexicon by using subsequences of
    the words and compound words.  Each word is shortened to
    max_word_len. Compound words are connected by a connector.
    Compound words longer than compound_word_step are shortened to
    that length for storage purposes.  All words except upper ontology
    labels are lower cased.  
    """
    if onto_name is None:
      onto_name = self.x_lingual_onto_name
    if onto_name == self.x_lingual_onto_name:
      self.x_lingual_lexicon_by_prefix = ontology = self.ontology[onto_name] = self.ontology.get(onto_name, self.x_lingual_lexicon_by_prefix)
    else:
      ontology = self.ontology[onto_name] = self.ontology.get(onto_name, {})
    if max_word_len is None: max_word_len =self.max_word_len
    compound_word_step = self.compound_word_step
    connector = self.connector
    if type(word2ner) is dict:
      word2ner = list(word2ner.items())
    lexicon = {}
    _max_lexicon = self._max_lexicon 
    for _idx, word_label in enumerate(word2ner):
      _idx += _max_lexicon
      word, label = word_label
      #if word.startswith('geor'): print (word, label)
      label = label.upper()
      is_cjk = self.cjk_detect(word)
      if is_cjk:
        word = "_".join(self.mt5_tokenizer.tokenize(word)).replace(mt5_underscore,"_").replace("__", "_").replace("__", "_").strip("_")
      word = word.strip().lower().translate(trannum).strip(self.strip_chars).replace(" ",connector)
      wordArr = word.split(connector)
      orig_lens = len(word) + len(wordArr)
      #wordArr = [w2.strip(self.strip_chars) for w2 in wordArr if w2.strip(self.strip_chars)]
      #print (word)
      # some proper nouns start with stop words like determiners. let's strip those.
      while wordArr:
        if wordArr[0] in self.stopwords:
          wordArr= wordArr[1:]
        else:
          break
      if not wordArr:
        continue
      word = connector.join(wordArr)
      #we don't have an actual count of the word in the corpus, so we create a weight based 
      #on the length, assuming shorter words with less compound parts are more frequent
      weight = 1/(1.0+math.sqrt(orig_lens))
      lenWordArr = len(wordArr)
      if lenWordArr == 0: 
        continue
      # add some randomness and only do suffix ends in some cases. TODO: we can use a config var.
      for wordArr in self._get_all_word_shingles(wordArr, max_word_len=max_word_len, create_suffix_end = _idx % 5 == 0):
        if not wordArr: continue
        word = connector.join(wordArr)
        key = (word, lenWordArr//(compound_word_step+1))
        #print (word0, word, weight)
        if type(label) in (list, tuple, set):
          if type(label) != list:
            label = list(label)
          _label, _idx, _cnt = lexicon.get(key, [label, _idx, {}])
          if _cnt is None: _cnt = {}
          _cnt[label[0]] = _cnt.get(label[0], 0.0) + weight
          lexicon[key] = [_label, _idx, _cnt]
        else:
          _label, _idx, _cnt = lexicon.get(key, [[label], _idx, {}])
          if _cnt is None: _cnt = {}
          _cnt[label] = _cnt.get(label, 0.0) + weight
          lexicon[key] = [_label, _idx, _cnt]
        prev_val= ontology.get(wordArr[0], [1, 100])
        ontology[wordArr[0]] = [max(lenWordArr, prev_val[0]), 2 if lenWordArr == 2 else min(max(lenWordArr-1,1), prev_val[1])]
    for key in lexicon:
      _cnt = lexicon[key][2]
      if _cnt:
        label = Counter(_cnt).most_common(1)[0][0]
        lexicon[key][0] = lexicon.get(label, [[label]])[0]
        lexicon[key] = lexicon[key][:-1]
    for word, slot in lexicon:
      prefix = word.split(connector,1)[0]
      if prefix in ontology:
        rec = ontology[prefix]
        if len(rec) == 2:
          rec.append({})
          rec.append({})
          rec.append({})
          rec.append({})
        lexicon2 = rec[2+min(3,slot)]
        if connector in word:
          word2 = '*'+connector+word.split(connector,1)[1]
        else:
          word2 = word
        lexicon2[word2] = lexicon[(word, slot)]
    self._max_lexicon += len(word2ner)

  def cjk_pre_tokenize(self, text, connector=None):
    """ tokenize using mt5. meant for cjk languages"""
    if connector is None:
      connector = self.connector
    if self.mt5_tokenizer is None:
      self.mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    words = self.mt5_tokenizer.tokenize(text.replace("_", " ").strip())
    words2 = []
    for word in words:
      if not words2: 
        words2.append(word)
        continue
      if not self.cjk_detect(word):
        if not self.cjk_detect(words2[-1]):
          if words2[-1] in self.strip_chars_set:
            words2[-1] += " "+word
          else:
            words2[-1] += word
          continue  
      words2.append(word)
    text = " ".join(words2).replace(mt5_underscore," ").replace("  ", " ").replace("  ", " ").strip()
    return text

  def in_ontology(self, word, connector=None, supress_cjk_tokenize=False, check_person_org_gpe_caps=True):
    """ find whether a word is in the ontology. """
    orig_word = word

    max_word_len =self.max_word_len
    min_word_len =self.min_word_len
    compound_word_step = self.compound_word_step
    if connector is None:
      connector = self.connector
    is_cjk = self.cjk_detect(word)
    if not is_cjk and len(word) < min_word_len:
      return word, None
    if not supress_cjk_tokenize and is_cjk:
      word = self.cjk_pre_tokenize(word, connector)
      
    word = word.strip().translate(trannum).strip(self.strip_chars+connector).replace(" ",connector)
    wordArr = word.split(connector) 
    if word in self.target_lang_lexicon:
      return orig_word, self.target_lang_lexicon[word]
    if is_cjk:
      word = word.replace(connector,"")
    if word in self.target_lang_lexicon: 
      return orig_word, self.target_lang_lexicon[word]
    #wordArr = [w2.strip(self.strip_chars) for w2 in wordArr if w2.strip(self.strip_chars)]
    if not wordArr or not wordArr[0] or not wordArr[-1]: 
      return word, None
    lenWordArr = len(wordArr)
    all_shingles = self._get_all_word_shingles(wordArr, max_word_len=max_word_len, create_suffix_end=not is_cjk)
    long_shingles= [] if is_cjk else self._get_all_word_shingles(wordArr, max_word_len=100000, create_suffix_end=False)
    for ontology in reversed(list(self.ontology.values())):
      #find patterned variations (shingles)
      for shingleArr in long_shingles + all_shingles: # we can probably dedup to make it faster
        if shingleArr and shingleArr[0] in ontology:
          lexicon2 = ontology[shingleArr[0]][2+min(3,lenWordArr//(compound_word_step+1))]
          if len(shingleArr) > 1:
            shingle = '*'+connector+connector.join((shingleArr[1:]))
          else:
            shingle = shingleArr[0]
          label, _ = lexicon2.get(shingle, (None, None))
          #let's return only labels that are in the upper_ontology
          if label is not None and (label[0] in self.upper_ontology or self.label2label.get(label[0]) in self.upper_ontology):
            if check_person_org_gpe_caps and ("PUBLIC_FIGURE" in label or "PERSON" in label or "ORG" in label or "GPE" in label):
              #ideally we would keep patterns like AaA as part of the shingle to match. This is a hack.
              if wordArr[0][0] != wordArr[0][0].upper() or  wordArr[-1][0] != wordArr[-1][0].upper(): continue
            label = label[0]
            label = self.label2label.get(label, label)
            return word, label
    return orig_word, None

  def _get_ngram_start_end(self, start_word):
    """ find the possible range of a compound word that starts with start_word """
    ngram_start = -1
    ngram_end = 100000
    for ontology in self.ontology.values():
      rec = ontology.get(start_word, [ngram_start, ngram_end])
      ngram_start, ngram_end = max(ngram_start,rec[0]), min(ngram_end,rec[1])
    return ngram_start, ngram_end
        

  def tokenize(self, text, connector=None, supress_cjk_tokenize=False, return_dict=True, row_id=0, doc_id=0):
    """
    Parse text for words in the ontology.  For compound words,
    transform into single word sequence, with a word potentially
    having a connector seperator.  Optionally, use the mt5 tokenizer
    to separate the words into subtokens first, and then do multi-word
    parsing.  Used for mapping a word back to an item in an ontology.
    Returns the tokenized text along with word to ner label mapping
    for words in this text.
    """
    max_word_len =self.max_word_len
    compound_word_step = self.compound_word_step
    labels = []
    if connector is None:
      connector = self.connector
    if not supress_cjk_tokenize and self.cjk_detect(text):
      text = self.cjk_pre_tokenize(text, connector)
    sent = text.strip().split()
    len_sent = len(sent)
    pos = 0
    for i in range(len_sent-1):
      if sent[i] is None: continue
      start_word =  sent[i].lower() #.strip(self.strip_chars) 
      if start_word in self.stopwords: 
        pos += len(sent[i])+1
        continue
      start_word = start_word.translate(trannum).split(connector)[0]
      start_word = start_word if len(start_word) <=  max_word_len else start_word[:max_word_len]
      ngram_start, ngram_end = self._get_ngram_start_end(start_word)
      #print (start_word, ngram_start, ngram_end)
      if ngram_start > 0:
        for j in range(ngram_start-1, ngram_end-2, -1):
          if len_sent - i  > j:
            wordArr = sent[i:i+1+j]
            new_word = " ".join(wordArr)
            if not self._has_nonstopword(wordArr): break
            # we don't match sequences that start and end with stopwords
            if wordArr[-1].lower() in self.stopwords: continue
            _, label = self.in_ontology(new_word, connector=connector, supress_cjk_tokenize=True)
            if label is not None:
              new_word = new_word.replace(" ", connector)
              new_word = new_word.lstrip(",")
              if new_word not in self.stopwords:
                #print ('found', new_word)
                sent[i] = new_word
                labels.append(((new_word, pos, pos + len(new_word), row_id, doc_id), label))
                for k in range(i+1, i+j+1):
                  sent[k] = None  
                break
      pos += len(sent[i])+1
    if return_dict:
      return {'text': " ".join([s for s in sent if s]), 'chunk2ner': dict(labels)}   
    else:
      return " ".join([s for s in sent if s]) 

  def cjk_detect(self, texts):
    # korean
    if re.search("[\uac00-\ud7a3]", texts):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", texts):
        return "ja"
    # chinese
    if re.search("[\u4e00-\u9FFF]", texts):
        return "zh"
    return None

if __name__ == "__main__":  
  data_dir = tmp_dir = None
  if "-s" in sys.argv:
    tmp_dir = sys.argv[sys.argv.index("-s")+1]
  if "-t" in sys.argv:
    sentence = sys.argv[sys.argv.index("-t")+1]
    manager = OntologyManager(data_dir=data_dir, tmp_dir=tmp_dir)
    txt = manager.tokenize(sentence)
    print(txt)

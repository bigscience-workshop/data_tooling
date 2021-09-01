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
  """
  Used to pre-process text. Users Spacy and Neuralcoref to obtain NER labels and coreference labels.
  
  Provides basic functionality for English NER tagging and chunking. 

  Recognizes the following  using Spacy NER:

      PERSON - People, including fictional.
      NORP - Nationalities or religious or political groups.
      FAC - Buildings, airports, highways, bridges, etc.
      ORG - Companies, agencies, institutions, etc.
      GPE - Countries, cities, states.
      LOC - Non-GPE locations, mountain ranges, bodies of water.
      PRODUCT - Objects, vehicles, foods, etc. (Not services.)
      EVENT - Named hurricanes, battles, wars, sports events, etc.
      WORK_OF_ART - Titles of books, songs, etc.
      LAW - Named documents made into laws.
      LANGUAGE - Any named language.
      DATE - Absolute or relative dates or periods.
      TIME - Times smaller than a day.
      PERCENT - Percentage, including ”%“.
      MONEY - Monetary values, including unit.
      QUANTITY - Measurements, as of weight or distance.
      ORDINAL - “first”, “second”, etc.
      CARDINAL - Numerals that do not fall under another type.

  """

  # basic template ontology that might be useful when parsing. the actual ontology is stored in the '.ontology' class variable.
  # an ontology maps a word to a parent hiearchy, with the top of the hiearchy being the last item.
  # leaves of the ontology are all lower cased, and intermediate leaves are upper case. 
  basic_ontology = {
      'PERSON': ['PERSON'],
      'NORP': ['NORP'],
      'FAC': ['FAC', 'PLACE'],
      'ORG': ['ORG'],
      'GPE': ['GPE', 'PLACE'],
      'LOC': ['LOC', 'PLACE'],
      'PRODUCT': ['PRODUCT', 'THING'],
      'EVENT': ['EVENT'],
      'WORK_OF_ART': ['WORK_OF_ART', 'THING'],
      'LAW': ['LAW', 'THING'],
      'LANGUAGE': ['LANGUAGE'],
      'DATE': ['DATE', 'DATE_AND_TIME'],
      'TIME': ['TIME', 'DATE_AND_TIME'],
      'PERCENT': ['PERCENT'],
      'MONEY': ['MONEY'],
      'QUANTITY': ['QUANTITY'],
      'ORDINAL': ['ORDINAL'],
      'CARDINAL': ['CARDINAL'],
  }

  nlp = None
  stopwords_en = {}
  default_strip_chars="-,~`!@#$%^&*(){}[]|\\/-_+=<>;:'\""
  default_person_pronouns = ("who", "whom", "whose", "our", "ours", "you", "your", "my", "i", "me", "mine", "he", "she", "his", "her", "him", "hers", "we")
  default_other_pronouns=("it", "its", "they", "their", "theirs", "them", "we")

  def __init__(self,  ner_regexes=None, ontology=None, strip_chars=None, person_pronouns=None,  other_pronouns=None, connector="_"):
    self.connector = connector
    if strip_chars is None:
      strip_chars = Processor.default_strip_chars
    if person_pronouns is None:
      person_pronouns = Processor.default_person_pronouns
    if other_pronouns is None:
      other_pronouns = Processor.default_other_pronouns
    self.strip_chars = strip_chars
    self.person_pronouns = person_pronouns
    self.other_pronouns = other_pronouns
    self.pronouns = set(list(person_pronouns)+list(other_pronouns))

    if not Processor.stopwords_en:
      Processor.stopwords_en = set(stopwords.words('english'))

    # we are storing the nlp object as a class variable to save on loading time. 
    if Processor.nlp is None:
      Processor.nlp = spacy.load('en_core_web_lg')
      #e.g., conv_dict={"Angela": ["woman", "girl"]}
      coref = neuralcoref.NeuralCoref(Processor.nlp.vocab) #, conv_dict
      Processor.nlp.add_pipe(coref, name='neuralcoref')
    if ontology is None: ontology = {}
    self.ontology = ontology
    for key, val in Processor.basic_ontology.items():
      self.ontology[key] = val
    if ner_regexes is None:
      ner_regexes = {}
    self.ner_regexes = ner_regexes

  def add_ontology(self, onto):
    for word, label in onto.items():
      label = label.upper()
      word = word.lower()
      self.ontology[word] = self.ontology.get(label, [label])
      word = word.strip(self.strip_chars)
      if len(word) > 5: word = word[:5]
      self.ontology[word] = self.ontology.get(label, [label])

  def analyze_with_ner_coref_en(self, text,  row_id=0, doc_id=0, chunk2ner=None, ref2chunk=None, chunk2ref=None, ontology=None, ner_regexes=None, connector=None):
    """
    Process NER on spans of text. Apply the coref clustering from neuralcoref. Use rules to expand and cleanup the coref and ner labeling.
    :arg text:
    :arg row_id
    :arg doc_id
    :arg chunk2ner
    :arg ref2chunk
    :arg chunk2ref
    :arg ontology
    :arg ner_regexes
    :arg connector
    
    Return a hash of form {'text': text, 'chunks':chunks, 'chunk2ner': chunk2ner, 'ref2chunk': ref2chunk, 'chunk2ref': chunk2ref}  
    Each chunks is in the form of a list of tuples [(text_span, start_id, end_id, doc_id, row_id), ...]
    A note on terminology. A span is a segment of text of one or more words. 
    A mention is a chunk that is recognized by some processor. 
    """
    
    def add_chunks_span(chunks, new_mention, old_mention, label, coref, chunk2ner, chunk2ref, ref2chunk):
      """ add a span to the chunks sequence and update the various ref and NER hashes """
      if old_mention in chunk2ner:
        del chunk2ner[old_mention]
      if label:
        chunk2ner[new_mention] = label
      if old_mention in chunk2ref:
        old_ref = chunk2ref[old_mention]
        ref2chunk[old_ref].remove(old_mention)
        if not ref2chunk[old_ref]:
          del ref2chunk[old_ref]
        del chunk2ref[old_mention]
      if new_mention in chunk2ref and coref != chunk2ref[new_mention]:
        old_ref = chunk2ref[new_mention]
        ref2chunk[old_ref].remove(new_mention)
        if not ref2chunk[old_ref]:
          del ref2chunk[old_ref]
        del chunk2ref[new_mention]
      if coref:
        chunk2ref[new_mention] = coref
        lst = ref2chunk.get(coref, [])
        if new_mention not in lst:
          ref2chunk[coref] = lst + [new_mention]
      chunks.append(new_mention)

    def del_ner_coref(old_mention, chunk2ner, chunk2ref, ref2chunk):
      """ remove an old_mention from the various NER and ref hashes """

      if old_mention in chunk2ner:
        del chunk2ner[old_mention]
      if old_mention in chunk2ref:
        old_ref = chunk2ref[old_mention]
        ref2chunk[old_ref].remove(old_mention)
        if not ref2chunk[old_ref]:
          del ref2chunk[old_ref]
        del chunk2ref[old_mention]

    #######
    if connector is None:
      connector = self.connector
    pronouns = self.pronouns
    person_pronouns = self.person_pronouns
    ret={}
    nlp = self.nlp
    if ref2chunk is None:
      ref2chunk = {}
    if chunk2ref is None:
      chunk2ref = {}
    if chunk2ner is None:
      chunk2ner = {}
    if ontology is None:
      ontology = self.ontology
    if ner_regexes is None:
      ner_regexes = self.ner_regexes
    text = text.strip()
    if not text: return ret
    if text[0] == '{' and text[-1] == '}':
      ret = json.loads(text)
      text = ret['text']
      doc_id = ret.get('doc_id', doc_id)
      row_id = ret.get('id', row_id)
      chunk2ner = ret.get('chunk2ner', chunk2ner)
      chunk2ref = ret.get('chunk2ref', chunk2ref)
      ref2chunk = ret.get('ref2chunk', ref2chunk)
    doc = nlp(text)

    #store away NOUNs for potential label and coref reference
    #rule for promotig a noun span into one considered for further processing:
    # - length of the number of words > 2 or length of span > 2 and the span is all uppercase (for abbreviations)
    # coref candidates:
    # - create an abbreviation from noun phrases as a candidate coref.
    # - use either the last two words a span as a candidate coref.
    # - use the abbreviation as a candidate coref
    for entity in list(doc.noun_chunks) + list(doc.ents):
      chunk2ner[(entity.text, entity.start, entity.end, row_id, doc_id)]= "NOUN"
      mention_lower = entity.text.lower()
      textArr = mention_lower.split()
      if len(textArr) > 2:
        short_span = " ".join(textArr[-2:])
        ref2chunk[short_span] = ref2chunk.get(short_span, []) + [(entity.text, entity.start, entity.end, row_id, doc_id)]
        non_stopwords = [a for a in textArr if a not in self.stopwords_en]
        if len(non_stopwords) > 2:
          abrev = "".join([a[0] for a in non_stopwords])
          ref2chunk[abrev] = ref2chunk.get(abrev, []) + [(entity.text, entity.start, entity.end, row_id, doc_id)]
      elif (len(entity.text) >=2 and entity.text == entity.text.upper()):
        ref2chunk[entity.text.lower()] = ref2chunk.get(entity.text.lower(), []) + [(entity.text, entity.start, entity.end, row_id, doc_id)]

    #store away coref NOUNs for potential label and coref reference
    #same rule as above for promoting a noun span into one considered for further processing.
    for cl in doc._.coref_clusters:
      mentions = [(entity.text, entity.start, entity.end, row_id, doc_id) for entity in cl.mentions]
      mentions.sort(key=lambda e: len(e[0]), reverse=True)
      textArr = mentions[0][0].lower().split()
      for key in mentions:
        chunk2ner[key]= "NOUN"
      for mention in mentions:
        mention_lower = mention[0].lower()
        textArr = mention_lower.split()
        if mention_lower not in self.stopwords_en:
          if len(textArr) > 1:
            short_span = " ".join(textArr[-2:])
          else:
            short_span = textArr[0]
          ref2chunk[short_span] = ref2chunk.get(short_span, []) + mentions
          non_stopwords = [a for a in textArr if a not in self.stopwords_en]
          if len(non_stopwords) > 2:
            abrev = "".join([a[0] for a in non_stopwords])
            ref2chunk[abrev] = ref2chunk.get(abrev, []) + mentions
    
    #cleanup the chunk2ref, favoring large clusters with coref labels that are longer
    seen = {}
    corefs = [(a, list(set(b))) for a, b in ref2chunk.items()]
    corefs.sort(key=lambda a: a[0].count(" ")+len(a[1]), reverse=True)
    for coref, spans in corefs:
      new_spans = []
      spans = list(set(spans))
      spans.sort(key=lambda a: a[1]+(1.0/(1.0+a[2]-a[1])))
      spans2 = []
      for span in spans:
        if spans2 and spans2[-1][1] >= span[1]:
          continue
        spans2.append(span)
      for span in spans2:
        if span in seen: continue
        seen[span] = 1
        new_spans.append(span)
      del ref2chunk[coref]
      if new_spans:
        new_coref = [s[0] for s in new_spans]
        new_coref.sort(key=lambda a: len(a), reverse=True)
        ref2chunk[new_coref[0].lower()] = list(set(list(ref2chunk.get(new_coref[0].lower(), [])) + new_spans))

    chunk2ref.clear()
    for a, b1 in ref2chunk.items():
      for b in b1:
        chunk2ref[b] = a
    
    # expand coref information by using the most common coref label in a cluster
    if doc._.has_coref:
      for cl in doc._.coref_clusters:
        mentions = [(entity.text, entity.start, entity.end, row_id, doc_id) for entity in cl.mentions]
        all_mentions = list(set(itertools.chain(*[ref2chunk[chunk2ref[mention]] for mention in mentions if mention in chunk2ref])))
        corefs = [chunk2ref[mention] for mention in mentions if mention in chunk2ref]
        if corefs:
          coref = Counter(corefs).most_common()[0][0]
        else:
          coref = cl.main.text.lower()
        for mention in all_mentions:
          if mention not in chunk2ner:
            chunk2ner[mention] = 'NOUN'
          old_ref = chunk2ref.get(mention)
          if old_ref and mention in ref2chunk[old_ref]:
            ref2chunk[old_ref].remove(mention)
            if not ref2chunk[old_ref]:
              del ref2chunk[old_ref]
          chunk2ref[mention] = coref
          if mention not in ref2chunk[coref]:
            ref2chunk[coref].append(mention)

    #expand ner labels based on coref matches 
    for entity in doc.ents:
      mention = (entity.text, entity.start, entity.end, row_id, doc_id)
      chunk2ner[mention]= entity.label_  
      if mention in chunk2ref:
        coref = chunk2ref[mention]
        for mention in ref2chunk[coref]:
          chunk2ner[mention] = entity.label_  

    # overwrite all ner labels in the coref cluster to PERSON if there is a person pronoun
    if doc._.has_coref:
      for cl in doc._.coref_clusters:
        cluster_text_list = set([m.text.lower() for m in cl.mentions])
        # I don't use "us" because that is sometimes the U.S.
        if "you" in cluster_text_list or "your"  in cluster_text_list  or "yours"  in cluster_text_list  or  "we" in cluster_text_list  or 'i' in cluster_text_list  or 'my' in cluster_text_list  or 'mine' in cluster_text_list or 'me' in cluster_text_list or 'he' in cluster_text_list or "she" in cluster_text_list or "his" in cluster_text_list or "her" in cluster_text_list or "him" in cluster_text_list or "hers" in cluster_text_list:
          label = "PERSON"
          for m in cl.mentions:
            chunk2ner[(m.text, m.start, m.end, row_id, doc_id)] = label
    
    # propogate the ner label to everything in the same coref group
    for coref, chunks in ref2chunk.items():
      labels = [chunk2ner[mention]  for mention in chunks if mention in chunk2ner and chunk2ner[mention] != 'NOUN']
      if labels:
        label = Counter(labels).most_common()[0][0]
        for mention in chunks:
          chunk2ner[mention] = label

    #add other words from the document into a sequence of form (word, start_idx, end_idx, row_id, doc_id)
    #add in coref label into the sequence
    #clear duplicates and subsumed mentions 
    chunks = [a for a in chunk2ner.items() if a[0][-2] == row_id and a[0][-1] == doc_id] 
    chunks.sort(key=lambda a: a[0][1]+(1.0/(1.0+a[0][2]-a[0][1])))
    chunks2 = []
    for mention, label in chunks:
      if not chunks2 or (chunks2[-1][2] <= mention[1]):
        if not chunks2 or chunks2[-1][2] < mention[1]: 
          add_chunks_span(chunks2, (doc[0 if not chunks2 else chunks2[-1][2]: mention[1]].text, 0 if not chunks2 else chunks2[-1][2], mention[1], row_id, doc_id), None, None, None, chunk2ner, chunk2ref, ref2chunk)
        add_chunks_span(chunks2, mention, None, label, chunk2ref.get(mention), chunk2ner, chunk2ref, ref2chunk)
      elif chunks2[-1][2] > mention[1] and chunks2[-1][1] <= mention[1]:
        if chunk2ner.get(chunks2[-1]) not in (None, '', 'NOUN'):
          del_ner_coref(mention, chunk2ner, chunk2ref, ref2chunk)
          continue
        elif label in  (None, '', 'NOUN'):
          del_ner_coref(mention, chunk2ner, chunk2ref, ref2chunk)
          continue
        old_mention = chunks2.pop()
        oldSpan = old_mention[0]
        oldLabel = chunk2ner.get(old_mention)
        oldAnaphore = chunk2ref.get(old_mention)
        sArr = oldSpan.split(mention[0], 1)
        del_ner_coref(old_mention, chunk2ner, chunk2ref, ref2chunk)
        s0 = sArr[0].strip()
        if s0:
          add_chunks_span(chunks2, (s0, old_mention[1], mention[1], row_id, doc_id), None, oldLabel if s0 in pronouns or (len(s0) > 1 and s0 not in self.stopwords_en) else None, oldAnaphore  if s0 in pronouns or (len(s0) > 1 and s0 not in self.stopwords_en) else None, chunk2ner, chunk2ref, ref2chunk)
        add_chunks_span(chunks2,  mention, None, label, oldAnaphore if not chunk2ref.get(mention) else chunk2ref.get(mention), chunk2ner, chunk2ref, ref2chunk)
        if len(sArr) > 1:
          s1 = sArr[1].strip()
          if s1:
            add_chunks_span(chunks2, (s1, mention[2], old_mention[2], row_id, doc_id), None,  oldLabel if s1 in pronouns or (len(s1) > 1 and s1 not in self.stopwords_en) else None, oldAnaphore  if s1 in pronouns or (len(s1) > 1 and s1 not in self.stopwords_en) else None, chunk2ner, chunk2ref, ref2chunk)
    len_doc = len(doc)
    if chunks2[-1][2] < len_doc:
      add_chunks_span(chunks2, (doc[chunks2[-1][2]:].text, chunks2[-1][2], len_doc, row_id, doc_id), None, None, None, chunk2ner, chunk2ref, ref2chunk)

    # propogate the ner label to everything in the same coref group
    for coref, chunks in ref2chunk.items():
      labels = [chunk2ner[mention]  for mention in chunks if mention in chunk2ner and chunk2ner[mention] != 'NOUN']
      if labels:
        label = Counter(labels).most_common()[0][0]
        for mention in chunks:
          chunk2ner[mention] = label

    #reset the indexes for to chunks to be per character index.
    #apply regex and ontology to add more labels
    #capture more coref for pronouns based on coref match of surronding info   
    chunks = []
    len_chunks2 = len(chunks2)
    for spanIdx, mention in enumerate(chunks2):
      label = chunk2ner.get(mention)
      coref = chunk2ref.get(mention)
      del_ner_coref(mention, chunk2ner, chunk2ref, ref2chunk)
      spanStr = mention[0]
      prev_words = []
      label2word = {}
      idx = 0
      # TODO, check surronding words for additional contexts when doing regexes
      for label2, regex in ner_regexes.items():
        if type(regex) is not list:
          regex = [regex]
        for regex0 in regex:
          for x in regex0.findall(spanStr):
            if type(x) != str: continue
            spanStr = spanStr.replace(x.strip(), " "+label2.upper()+str(idx)+" ")
            label2word[label2.upper()+str(idx)] = (x.strip(), label2)
            idx += 1
          
      # TODO, do multiword like "young woman" and patterns like "old* woman", or "*ician"
      spanArr = spanStr.split()
      for idx_word, orig_word in enumerate(spanArr):
        if orig_word in label2word:
          if prev_words:
            pWords = [w.strip(self.strip_chars) for w in prev_words]
            pWords = [w if len(w) <= 5 else w[:5] for w in pWords if w]
            pWords = connector.join(pWords)
            new_word = " ".join(prev_words)
            len_new_word = len(new_word)
            add_chunks_span(chunks, (new_word, 0 if not chunks else chunks[-1][2]+1,  len_new_word if not chunks else chunks[-1][2]+1+len_new_word, row_id, doc_id), None, ontology.get(pWords, [label])[0], coref, chunk2ner, chunk2ref, ref2chunk)
          new_word = label2word[orig_word][0]
          len_new_word = len(new_word)
          add_chunks_span(chunks, (new_word, 0 if not chunks else chunks[-1][2]+1,  len_new_word if not chunks else chunks[-1][2]+1+len_new_word, row_id, doc_id), None, label2word[orig_word][1], coref, chunk2ner, chunk2ref, ref2chunk)
          prev_words = []
          continue
        word = orig_word.lower()
        if not coref and word in pronouns:
          for idx in (spanIdx-1, spanIdx-2, spanIdx+1 , spanIdx+2 ):
            if idx >= 0 and idx < len_chunks2  and chunk2ref.get(chunks2[idx]):
              coref = chunk2ref.get(chunks2[idx])
              break
        if not coref and word in pronouns:
          for idx in (spanIdx-1, spanIdx-2, spanIdx+1 , spanIdx+2 ):
            if idx >= 0 and idx < len_chunks2  and chunk2ner.get(chunks2[idx]) and chunks2[idx][0].lower() not in pronouns:
              if word in person_pronouns and 'PERSON' not in self.ontology.get(chunk2ner.get(chunks2[idx]), []): 
                continue
              coref = chunk2ref.get(chunks2[idx])
              break
        if word in ontology:
          if prev_words:
            pWords = [w.strip(self.strip_chars) for w in prev_words]
            pWords = [w if len(w) <= 5 else w[:5] for w in pWords if w]
            pWords = connector.join(pWords)
            new_word = " ".join(prev_words)
            len_new_word = len(new_word)
            add_chunks_span(chunks, (new_word, 0 if not chunks else chunks[-1][2]+1,  len_new_word if not chunks else chunks[-1][2]+1+len_new_word, row_id, doc_id), None, ontology.get(pWords, [label])[0], coref, chunk2ner, chunk2ref, ref2chunk)
          len_new_word = len(mention[0])
          add_chunks_span(chunks, (mention[0], 0 if not chunks else chunks[-1][2]+1,  len_new_word if not chunks else chunks[-1][2]+1+len_new_word, row_id, doc_id), None, ontology[word][0], coref, chunk2ner, chunk2ref, ref2chunk)
          prev_words = []
          continue

        word = word.strip(self.strip_chars)
        if len(word) > 5: word = word[:5]
        if word in ontology:
          if prev_words:
            pWords = [w.strip(self.strip_chars) for w in prev_words]
            pWords = [w if len(w) <= 5 else w[:5] for w in pWords if w]
            pWords = connector.join(pWords)
            new_word = " ".join(prev_words)
            len_new_word = len(new_word)
            add_chunks_span(chunks, (new_word, 0 if not chunks else chunks[-1][2]+1,  len_new_word if not chunks else chunks[-1][2]+1+len_new_word, row_id, doc_id), None, ontology.get(pWords, [label])[0], coref, chunk2ner, chunk2ref, ref2chunk)
          len_new_word = len(mention[0])
          add_chunks_span(chunks, (mention[0], 0 if not chunks else chunks[-1][2]+1,  len_new_word if not chunks else chunks[-1][2]+1+len_new_word, row_id, doc_id), None, ontology[word][0], coref, chunk2ner, chunk2ref, ref2chunk)
          prev_words = []
          continue
        prev_words.append(orig_word)

      if prev_words:
        pWords = [w.strip(self.strip_chars) for w in prev_words]
        pWords = [w if len(w) <= 5 else w[:5] for w in pWords if w]
        pWords = connector.join(pWords)
        new_word = " ".join(prev_words)
        len_new_word = len(new_word)
        add_chunks_span(chunks, (new_word, 0 if not chunks else chunks[-1][2]+1, len_new_word if not chunks else chunks[-1][2]+1+len_new_word, row_id, doc_id), None, ontology.get(pWords, [label])[0], coref, chunk2ner, chunk2ref, ref2chunk)
    
    ret['doc_id'] = doc_id
    ret['id'] = row_id
    ret['text'] = " ".join([c[0] for c in chunks])
    ret['chunks'] = chunks
    ret['chunk2ner'] = chunk2ner
    ret['chunk2ref'] = chunk2ref
    ret['ref2chunk'] = ref2chunk
    return ret

  def process(self, text="", batch=None, *args, **argv):
    """
    Process a single row of text or a batch. Performs basic chunking into roughly interesting phrases, and corefrence and NER identification.
    """
    # need paramater to decide if we want toreset the varioous *2* hashes for each example in a batch?
    row_id=argv.get('row_id',0)
    doc_id=argv.get('doc_id',0)
    chunk2ner=argv.get('chunk2ner')
    ref2chunk=argv.get('ref2chunk')
    chunk2ref=argv.get('chunk2ref')
    ontology=argv.get('ontology')
    ner_regexes=argv.get('ner_regexes')
    connector=argv.get('connector', "_")
    if batch is not None:
      return [self.analyze_with_ner_coref_en(text, row_id=row_id, doc_id=doc_id, chunk2ner=chunk2ner, ref2chunk=ref2chunk, chunk2ref=chunk2ref, ontology=ontology, ner_regexes=ner_regexes, connector=connector)  for text in batch]
    return self.analyze_with_ner_coref_en(text, row_id=row_id, doc_id=doc_id, chunk2ner=chunk2ner, ref2chunk=ref2chunk, chunk2ref=chunk2ref, ontology=ontology, ner_regexes=ner_regexes, connector=connector) 


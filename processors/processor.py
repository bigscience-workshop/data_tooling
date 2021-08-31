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



# TODO, add data lineage and processing meta-data, inlcuding processor state varaibles like random seeds.
# text examples can be processed and merged, and split. 
# we want to be able to identify a particular text example for removal or recompute based on 
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
      ret =  processor.process(batch=ret) #TODO, pass in doc_id and row_id/id ?
    return ret


class Processor:
  """
  Used to pre-process text. Provides basic functionality for English NER tagging and chunking. 

  Recognizes the following Spacy NER:

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

  def __init__(self,  ner_regexes=None, ontology=None):

    #TODO: we need to get the title and pronoun list for a particular language in order to do better backtrans matching
    if Processor.stopwords_en is {}:
      Processor.stopwords_en = set(stopwords.words('english'))

    # we are storing the nlp object as a class variable to save on loading time. 
    if Processor.nlp is None:
      Processor.nlp = spacy.load('en_core_web_lg')
      #e.g., conv_dict={"Angela": ["woman", "girl"]}
      coref = neuralcoref.NeuralCoref(Processor.nlp.vocab) #, conv_dict
      Processor.nlp.add_pipe(coref, name='neuralcoref')
    if ontology is None: ontology = {}
    self.ontology = ontology
    if ner_regexes is None:
      ner_regexes = {}
    self.ner_regexes = ner_regexes

  def add_ontology(self, onto):
    for word, label in onto.items():
      word = word.lower()
      self.ontology[word] = label
      word = word.strip("-,~`!@#$%^&*(){}[]|\\/-_+=<>;:'\"")
      if len(word) > 5: word = word[:5]
      self.ontology[word] = label

  nlp = None
  stopwords_en = {}
  

  def analyze_with_ner_coref_en(self, text, row_id=0, doc_id=0, chunk2ner=None, ref2chunk = None, chunk2ref=None, ontology=None, ner_regexes=None, connector="_", pronouns=("who", "whom", "whose", "our", "ours", "you", "your", "my", "i", "me", "mine", "he", "she", "his", "her", "him", "hers", "it", "its", "they", "their", "theirs", "them", "we")):
    """
    Process NER on spans of text. Apply the coref clustering from neuralcoref. 
    Use some rules to expand and cleanup the coref and ner labeling.
    Return a hash of form {'text': text, 'chunks':chunks, 'chunk2ner': chunk2ner, 'ref2chunk': ref2chunk, 'chunk2ref': chunk2ref}  
    Each chunks is in the form of a list of tuples [(text_span, start_id, end_id, doc_id, row_id), ...]
    A note on terminology. A span is a segment of text of one or more words. 
    A mention is a chunk that is recognized by some processor. 
    """

    def add_chunks_span(chunks, span, old_mention, label, coref, chunk2ner, chunk2ref, ref2chunk, row_id=0, doc_id=0):
      """ add a span to the chunks sequence and update the various ref and NER hashes """
      len_span = len(span)
      if not chunks:
        span_pos = 0
      else:
        span_pos = chunks[-1][2]+1
      new_mention = (span, span_pos, span_pos+len_span-1, row_id, doc_id)
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
      if coref:
        chunk2ref[new_mention] = coref
        ref2chunk[coref] = ref2chunk.get(coref, []) + [new_mention]
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
    for entity in list(doc.noun_chunks) + list(doc.ents):
      textArr = entity.text.lower().split()
      chunk2ner[(entity.text, entity.start, entity.end, row_id, doc_id)]= "NOUN"
      if len(textArr) > 2:
        short_span = " ".join(textArr[-2:]).lower()
        abrev = "".join([a[0] for a in textArr if a not in self.stopwords_en]).lower()
        ref2chunk[short_span] = ref2chunk.get(short_span, []) + [(entity.text, entity.start, entity.end, row_id, doc_id)]
        if len(abrev) > 2: ref2chunk[abrev] = ref2chunk.get(abrev, []) + [(entity.text, entity.start, entity.end, row_id, doc_id)]
      elif (len(entity.text) >=2 and entity.text == entity.text.upper()): # or (len(entity.text) >=4 and entity.text not in pronouns):
        ref2chunk[entity.text.lower()] = ref2chunk.get(entity.text.lower(), []) + [(entity.text, entity.start, entity.end, row_id, doc_id)]
    
    #store away coref NOUNs for potential label and coref reference
    #same rule as above for promoting a noun span into one cosndiered for further processing.
    for cl in doc._.coref_clusters:
      mentions = [(entity.text, entity.start, entity.end, row_id, doc_id) for entity in cl.mentions]
      mentions.sort(key=lambda e: len(e[0]), reverse=True)
      textArr = mentions[0][0].lower().split()
      for key in mentions:
        chunk2ner[key]= "NOUN"
      for mention in mentions:
        textArr = mention[0].lower().split()
        if len(textArr) > 2:
          short_span = " ".join(textArr[-2:]).lower()
          abrev = "".join([a[0] for a in textArr if a not in self.stopwords_en]).lower()
          ref2chunk[short_span] = ref2chunk.get(short_span, []) + mentions
          if len(abrev) > 2: ref2chunk[abrev] = ref2chunk.get(abrev, []) + mentions
        elif (len(entity.text) >=2 and entity.text == entity.text.upper()): # or (len(mention[0]) >=4 and mention[0] not in pronouns):
          ref2chunk[mention[0].lower()] = ref2chunk.get(mention[0].lower(), []) + mentions
    
    #cleanup the chunk2ref, favoring large clusters
    seen = {}
    corefs = list(ref2chunk.items())
    corefs.sort(key=lambda a: len(a[1]), reverse=True)
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

    # expand coref information
    if doc._.has_coref:
      for cl in doc._.coref_clusters:
        mentions = [(entity.text, entity.start, entity.end, row_id, doc_id) for entity in cl.mentions]
        corefs = [chunk2ref[mention] for mention in mentions if mention in chunk2ref]
        if corefs:
          coref = Counter(corefs).most_common()[0][0]
          for mention in mentions:
            chunk2ref[mention] = coref
            if mention not in ref2chunk[coref]:
              ref2chunk[coref].append(mention)
        else:
          coref = cl.main.text.lower()
          for mention in mentions:
            chunk2ref[mention] = coref
            if coref not in ref2chunk:
              ref2chunk[coref] = []
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
    for coref, spans in ref2chunk.items():
      labels = [chunk2ner[mention]  for mention in spans if mention in chunk2ner and chunk2ner[mention] != 'NOUN']
      if labels:
        label = Counter(labels).most_common()[0][0]
        for mention in spans:
          chunk2ner[mention] = label

    #add other words from the document into a sequence of form (word, start_idx, end_idx, docId)
    #add in coref label into the sequence
    #clear duplicates and subsumed mentions 
    chunks = [a for a in chunk2ner.items() if a[0][-2] == row_id and a[0][-1] == doc_id]
    chunks.sort(key=lambda a: a[0][1]+(1.0/(1.0+a[0][2]-a[0][1])))
    chunks2 = []
    prevPos = 0
    for mention, label in chunks:
      if prevPos<= mention[1]:
        add_chunks_span(chunks2, doc[prevPos: mention[1]].text, mention, None, None, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
      else:
        if chunks2 and chunk2ner.get(chunks2[-1]) not in (None, '', 'NOUN'):
          del_ner_coref(mention, chunk2ner, chunk2ref, ref2chunk)
          continue
        elif label in  (None, '', 'NOUN'):
          del_ner_coref(mention, chunk2ner, chunk2ref, ref2chunk)
          continue
        oldSpan = chunks2[-1][0]
        oldLabel = chunks2[-1][1]
        oldAnaphore = chunks2[-1][2]
        sArr = oldSpan.split(mention[0], 1)
        old_mention = chunks2.pop()
        del_ner_coref(old_mention, chunk2ner, chunk2ref, ref2chunk)
        if sArr[0].strip():
          add_chunks_span(chunks2,  sArr[0].strip(), mention, oldLabel, oldAnaphore, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
        add_chunks_span(chunks2,  mention[0], mention, label, chunk2ref.get(mention), chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
        if sArr[1].strip():
          add_chunks_span(chunks2,  sArr[1].strip(), mention, oldLabel, oldAnaphore, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
        continue
      prevPos = mention[2]
      add_chunks_span(chunks2,  mention[0], mention, label, chunk2ref.get(mention), chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
    len_doc = len(doc)
    if prevPos < len_doc:
      add_chunks_span(chunks2,  doc[len_doc:].text, None, None, None, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
    
    #apply ontology to add more labels
    #capture more coref for pronouns based on coref match of surronding info   
    chunks = []
    len_chunks2 = len(chunks2)
    for spanIdx, mention in enumerate(chunks2):
      label, coref = chunk2ner.get(mention), chunk2ref.get(mention)
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
            print (x)
            spanStr = spanStr.replace(x.strip(), " "+label2.upper()+str(idx)+" ")
            label2word[label2.upper()+str(idx)] = (x.strip(), label2)
            idx += 1
            
      # TODO, do multiword like "young woman" and patterns like "old* woman", or "*ician"
      spanArr = spanStr.split()
      for idx_word, orig_word in enumerate(spanArr):
        if orig_word in label2word:
          if prev_words:
            pWords = [w.strip("-,~`!@#$%^&*(){}[]|\\/-_+=<>;:'\"") for w in prev_words]
            pWords = [w if len(w) <= 5 else w[:5] for w in pWords if w]
            pWords = connector.join(pWords)
            add_chunks_span(chunks, " ".join(prev_words), mention, ontology.get(pWords, label), coref, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
          add_chunks_span(chunks, label2word[orig_word][0], mention, label2word[orig_word][1], coref, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
          prev_words = []
          continue
        word = orig_word.lower()
        if not coref and word in pronouns:
          for idx in (spanIdx-1, spanIdx-2, spanIdx+1 , spanIdx+2 ):
            if idx >= 0 and idx < len_chunks2  and chunks2[idx][2]:
              coref = chunks2[idx][2]
              break
        if not coref and word in pronouns:
          for idx in (spanIdx-1, spanIdx-2, spanIdx+1 , spanIdx+2 ):
            if idx >= 0 and idx < len_chunks2  and chunks2[idx][2] and chunks2[idx][1] == 'NOUN' and chunks2[idx][0].lower() not in pronouns:
              coref = chunks2[idx][0].lower()
              break
        if word in ontology:
          if prev_words:
            pWords = [w.strip("-,~`!@#$%^&*(){}[]|\\/-_+=<>;:'\"") for w in prev_words]
            pWords = [w if len(w) <= 5 else w[:5] for w in pWords if w]
            pWords = connector.join(pWords)
            add_chunks_span(chunks, " ".join(prev_words), mention, ontology.get(pWords, label), coref, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
          add_chunks_span(chunks, orig_word, mention, ontology[word], coref, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
          prev_words = []
          continue

        word = word.strip("-,~`!@#$%^&*(){}[]|\\/-_+=<>;:'\"")
        if len(word) > 5: word = word[:5]
        if word in ontology:
          if prev_words:
            pWords = [w.strip("-,~`!@#$%^&*(){}[]|\\/-_+=<>;:'\"") for w in prev_words]
            pWords = [w if len(w) <= 5 else w[:5] for w in pWords if w]
            pWords = connector.join(pWords)
            add_chunks_span(chunks, " ".join(prev_words), mention, ontology.get(pWords, label), coref, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
          add_chunks_span(chunks, orig_word, mention, ontology[word], coref, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)
          prev_words = []
          continue
        prev_words.append(orig_word)

      if prev_words:
        pWords = [w.strip("-,~`!@#$%^&*(){}[]|\\/-_+=<>;:'\"") for w in prev_words]
        pWords = [w if len(w) <= 5 else w[:5] for w in pWords if w]
        pWords = connector.join(pWords)
        add_chunks_span(chunks, " ".join(prev_words), mention, ontology.get(pWords, label), coref, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)

    ret['doc_id'] = doc_id
    ret['id'] = row_id
    ret['text'] = " ".join([c[0] for c in chunks])
    ret['chunks'] = chunks
    ret['chunk2ner'] = chunk2ner
    ret['chunk2ref'] = chunk2ref
    ret['ref2chunk'] = ref2chunk
    return ret

  def process(self, text="", batch=None, *args, **argv):
    if batch is not None:
      return [self.analyze_with_ner_coref_en(text)  for text in batch]
    return self.analyze_with_ner_coref_en(text) #TODO, pass in doc_id and row_id/id ?
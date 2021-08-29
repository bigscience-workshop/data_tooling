from random import sample
import glob, os
import multiprocessing
from transformers import AutoTokenizer
import gzip
import fsspec, aiohttp, os, argparse
import itertools
from collections import Counter, OrderedDict
import os
import json
from transformers import AutoTokenizer
import threading
import numpy as np
import os
import time
    
import json
import copy
import fasttext
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from time import time
import numpy as np
from datasets import load_dataset
from collections import Counter
from itertools import chain
from bounter import bounter
import os
from joblib import dump, load
from joblib import Parallel, delayed
import glob
import os
import json
import math, os

mt5_underscore= "▁"

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir, os.path.pardir)))

from data_tooling.processors.processor import Processor



class OntologyProcessor(Processor):

  """ Gets preliminary labels using conceptnet for people, location, etc. """

  def init_coneptnet_data(self, arg):
    shared_dir = arg.shared_dir
    if not os.path.exists("./conceptnet-assertions-5.7.0.csv"):
      if not os.path.exists(f"{shared_dir}/conceptnet-assertions-5.7.0.csv.gz"):
        os.system(f"wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz")
        os.system(f"cp conceptnet-assertions-5.7.0.csv.gz {shared_dir}")
      else:
        os.system(f"cp {shared_dir}/conceptnet-assertions-5.7.0.csv.gz ./")
      os.system("gunzip ./conceptnet-assertions-5.7.0.csv.gz")
    
  def create_wn_cat(self, arg,  keep_percentage=.01):
    """
    extract linkage from conceptnet words and wordnet category words
    """
    shared_dir = arg.shared_dir
    if not os.path.exists(f"{shared_dir}/wn.csv"):
      os.system("grep '\/wn\/' conceptnet-assertions-5.7.0.csv > wn.csv")
      os.system(f"mv wn.csv {shared_dir}")
    wn =  open(f"{shared_dir}/wn.csv", "rb").read().decode().split('\n')
    wn = [s.split('\t')[0] for s in wn]
    wn = [s.strip(']').split(',/c/')[1:] for s in wn]

    wn_list = itertools.chain(*[[w.split("/")[1] if w.split("/")[-2] == "wn" else w.split("/")[-2] for w in awn if "_" in w] for awn in wn])
    items = list(Counter([w for w in wn_list if w[0] not in "0123456789"]).items())
    items.sort(key=lambda a:a[1], reverse=True)
    items = [i for i in items if i[1] != 1]
    typeHash = dict( items[:int(len(items)*keep_percentage)])
    wn_list2 = itertools.chain(*[[w for w in awn if (w.split("/")[2] == 'n') and (w.split("/")[0] != 'en') and ("_" in  w.split("/")[1]) and (w.split("/")[-2] in typeHash)] for awn in wn])
    items2 = list(Counter([w for w in wn_list2 if w[0] not in "0123456789"]).items())
    items2.sort(key=lambda a:a[1], reverse=True)
    return items2, typeHash

  def create_rel(self, arg):
    """
    extract words that are related to each other based on conceptnet
    """
    shared_dir = arg.shared_dir
    init_coneptnet_data(arg)
    if not os.path.exists(f"{shared_dir}/syn.csv"):
        os.system(f"grep '\/r\/Synonym\/' conceptnet-assertions-5.7.0.csv > syn.csv")
        os.system(f"grep 'SimilarTo\/' conceptnet-assertions-5.7.0.csv > sim.csv")
        os.system(f"grep 'MannerOf\/' conceptnet-assertions-5.7.0.csv > manner.csv")
        os.system(f"grep 'DistinctFrom\/' conceptnet-assertions-5.7.0.csv > dest.csv")
        os.system(f"grep 'DerivedFrom\/' conceptnet-assertions-5.7.0.csv > deri.csv")
        os.system(f"grep 'Antonym\/' conceptnet-assertions-5.7.0.csv > anti.csv")
        os.system(f"grep 'EtymologicallyRelatedTo\/' conceptnet-assertions-5.7.0.csv > erel.csv")
        os.system(f"grep 'EtymologicallyDerivedFrom\/' conceptnet-assertions-5.7.0.csv > ederi.csv")
        os.system(f"grep 'RelatedTo\/' conceptnet-assertions-5.7.0.csv > rel.csv")
        os.system(f"grep 'FormOf\/' conceptnet-assertions-5.7.0.csv > formof.csv")
        os.system(f"grep 'IsA\/' conceptnet-assertions-5.7.0.csv > isa.csv")
        os.system(f"mv syn.csv sim.csv manner.csv dest.csv deri.csv anti.csv erel.csv ederi.csv rel.csv formof.csv isa.csv {shared_dir}")
    rel2 = OrderedDict()
    for rel_type in ('syn', 'sim', 'manner', 'dest', 'deri', 'anti', 'erel', 'ederi', 'rel', 'formof','isa') :
      i = 0
      rel =  open(f"{shared_dir}/{rel_type}.csv", "rb").read().decode().split('\n')
      rel = [s.split('\t')[0] for s in rel]
      rel = [s.strip(']').split(',/c/')[1:] for s in rel]
      for s in rel:
        if len(s) < 2:
          continue
        a = s[0]
        b = s[1]
        a = a.split('/')
        lang1 = a[0]
        a = a[1]
        b = b.split('/')
        lang2 = b[0]
        b = b[1]
        if a == b:
          continue
        val = [a,b]
        if len(a) > len(b):
          if a in rel2:
            val = rel2[a] + [a,b]
            del rel2[a]
          rel2[b]  =  rel2.get(b, []) + val
        else:
          if b in rel2:
            val = rel2[b] + [a,b]
            del rel2[b]
          rel2[a]  =  rel2.get(a, []) +  val
        i+= 1
        #if i > 10000:
        #  break

      for key in list(rel2.keys()):
        rel2[key] = list(set(rel2[key]))

    return rel2


  def read_vec_file(self, arg, file, chunk_size = 5000000):
    """
    convert a ".vec" file into a memmap file and a label2id dict.
    """
    def reader(mmap_file, label2id, lines, start, end, vec_size, start_time):
      dat = []
      for idx, line in zip(range(start, end), lines):
        parts = line.rstrip().split(" ")
        if len(parts) == vec_size + 1:
          label2id[parts[0]] = idx
          dat.append(np.array([np.float32(x) for x in parts[1:]]))
        else:
          label2id[parts[0]] = idx
          dat.append(np.array([np.float32(x) for x in parts[1:]]+[0.0]*( vec_size + 1 - len(parts))))
          print ("WARNING: got a line without enough data at ", (parts[0], idx, len(parts) ))
      mmap_file[start:end] = np.vstack(dat)
      #do a lock or use logger
      #print (end, "items/s", end/(time.time()-start_time))    
    
    start_time = time.time()
    curr_idx = 0  
    label2id = {}
    size = os.path.getsize(file)
    workers=[]
    with open(file, "r") as f:
      num_vec, vec_size = [int(s) for s in f.readline().strip().split()]
      mmap_file = np.memmap(file.replace(".vec", ".mmap"), mode="w+", dtype=np.float32, shape=(num_vec, vec_size))
      while f.tell() < size:
        pos = f.tell()
        read_len = min(pos + chunk_size, size) - pos
        arr = f.read(read_len)
        if arr[-1] != '\n':
          arr += f.readline().rstrip()
        arr = arr.split("\n")
        if arr[0] == '':
          arr = arr[1:]
        if arr[-1] == '':
          arr = arr[:len(arr)-1]
        num_rec = len(arr)
        worker = threading.Thread(target=reader, args=(mmap_file, label2id, arr, curr_idx, curr_idx+num_rec, vec_size, start_time))
        workers.append(worker)
        worker.start()
        curr_idx += num_rec
      for worker in workers:
      worker.join()
      return label2id, file.replace(".vec", ".mmap")

  def shrink_numberbatch_data(self, arg):
    """
    Convert the larger numberbatch file into a smaller version, collapsing multilingual concepts into words
    """
    shared_dir = arg.shared_dir
    if not os.path.exists("./numberbatch.mmap "):
      if not os.path.exists(f"{shared_dir}/numberbatch.mmap"):
        os.system(f"wget https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz")
        os.system(f"gunzip numberbatch-19.08.txt.gz")
        os.system(f"mv numberbatch-19.08.txt numberbatch.vec")
        label2id,  file_name= read_vec_file("numberbatch.vec")
        os.system(f"cp {file_name} {shared_dir}")
        file_name=file_name.replace(".mmap", ".label2id")
        json.dump(label2id, open(file_name, "w", encoding="utf-8"))
        os.system(f"cp {file_name} {shared_dir}")
        id2label = list(label2id.items())
        id2label.sort(key= lambda x: x[1])
        id2label = [item[0] for item in id2label]
        file_name=file_name.replace(".label2id", ".id2label")
        json.dump(id2label, open(file_name, "w", encoding="utf-8"))
        os.system(f"cp {file_name} {shared_dir}")
        os.system(f"rm ./numberbatch.vec")
      os.system(f"cp {shared_dir}/numberbatch.mmap ./")
      os.system(f"cp {shared_dir}/numberbatch.id2label ./")
      os.system(f"cp {shared_dir}/numberbatch.label2id ./")
      
    id2label = json.load(open("./numberbatch.id2label"))
    label2id = json.load(open("./numberbatch.label2id"))
    numberbatch = np.memmap("./numberbatch.mmap", mode="r+", dtype=np.float32, shape=(len(label2id), 300))
    labelHash = OrderedDict()
    # collapse all words to remove "/c/lang/..." and replace "#" with 1
    for idx, label in enumerate(label2id):
      labelArr=label.split("/")
      mult=1
      if labelArr[2] == 'en':
        mult = 3
      label = labelArr[-1].replace("#", "1")
      labelHash[label] = labelHash.get(label, [])+  [idx]*mult

    found, missing = align_mt5_with_numberbatch(labelHash)
    rel = create_rel
    # create the collapsed form
    numberbatch2 = np.memmap("./numberbatch2.mmap", mode="w+", dtype=np.float32, shape=(len(labelHash), 300))
    id2label2 = list(labelHash.keys()) # assume it is ordered
    label2id2 = dict([(l, idx) for idx, l in enumerate(id2label2)])
    foundIdx=[label2id2[label] for label in found]

    for idx, l in enumerate(id2label2):
      numberbatch2[idx] = numberbatch[labelHash[l]].mean(axis=0)

    return id2label2, label2id2, numberbatch2, labelHash, foundIdx

  def align_mt5_with_numberbatch(self, arg, labelHash):
    """
    Find overlap between mt5 and numberbatch tokens so we can project new tokens from numberbatch back to mt5
    """
    mt5_tok = AutoTokenizer.from_pretrained("google/mt5-small")
    mt5_hash = OrderedDict()
    for b in range(len(mt5_tok)):
      a = mt5_tok.convert_ids_to_tokens(b)
      if a.strip("▁").lower():
        mt5_hash[a.strip("▁").lower()] = mt5_hash.get(a.strip("▁").lower(), []) +  [b]
    missing = []
    found = []
    for word in mt5_hash:
      word = word.replace("2", "1").replace("3", "1").replace("4", "1").replace("5", "1").replace("6", "1").replace("7", "1").replace("8", "1").replace("9", "1").replace("0", "1")
      if word not in labelHash:
        missing.append(word)
      else:
        found.append(word)
    
    return found, missing, mt5_file, mt5_hash

  def cluster_numberbatch(self, arg, file="numberbatch.mmap",  vocab_file="numberbatch.id2label", vec_size=300, words_per_cluster=10):
    if True:
      terms = json.load(open(vocab_file))
      num_vec = len(terms)
      x = np.memmap(file, mode="r+", dtype=np.float32, shape=(num_vec, vec_size))
      x = np.copy(x)
      true_k=int(num_vec/words_per_cluster)
      print (x)
      #k-means with word_vectors
      #TODO: if the embeddings can't fit in memory, mmemmap and https://ml.dask.org/modules/generated/dask_ml.cluster.KMeans.html ??
      km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                              init_size=max(true_k*3,1000), batch_size=1000).fit(x)
      dump(km, f'numberbatch.joblib') 
    cluster = {}
    for term, label in zip(terms, km.labels_):
      cluster[label] = cluster.get(label, [])+[term]
    return cluster
    

  def create_cn_ontology(self, arg):
    shared_dir = arg.shared_dir
    categories, typeHash = create_wn_cat(1.0)
    rel = create_rel()
    word2wncat = OrderedDict()
    #get a word to category mapping
    for concept, ct in categories:
      conceptArr = concept.split("/")
      word, wncat = conceptArr[1], conceptArr[-2]
      if wncat == 'linkdef':
        continue
      if word in word2wncat:
        if wncat != word2wncat[word]:
          word2wncat[word]='*'
      else:
        word2wncat[word] = wncat

    rel2=OrderedDict()
    rel3=OrderedDict()
    #group by categories, while expanding the categories
    for key, val in rel.items():
      if len(val) < 6:
        val2 = [(word2wncat[v] if v in word2wncat else (word2wncat[v.split("_")[0]] if v.split("_")[0] in word2wncat else word2wncat[v.split("_")[-1]]),v) for v in val if (v in word2wncat) or (v.split("_")[0] in word2wncat) or (v.split("_")[-1] in word2wncat)]
        if  val2: 
          val3 = [v for v in val if v not in val2]
          itr = itertools.groupby(val2, lambda x : x[0])
          groups = [(key, list(group)) for key, group in itr]
          groups.sort(key=lambda a: len(a[1]))
          max_cat = groups[-1][0]

          if max_cat != '*':
            # infer category of other words in this group if majority of labels is max_cat
            if len(groups[-1][1])*2 >= len(val):
              for word in val3:
                if word in word2wncat:
                  if wncat != word2wncat[word]:
                    word2wncat[word]='*'
                  else:
                    groups[-1][1].append((max_cat, word))
                else:
                  word2wncat[word] = max_cat
                  groups[-1][1].append((max_cat, word))
          all = {}
          for key, group in groups:
            if key == '*': continue
            group= list(group)
            if len(group) == 1:
              continue
            group = [g[1] for g in group]
            group.sort(key=lambda s: len(s))
            rel2[group[0]] = list(set(rel2.get(group[0],[]) + group))
            for g in group:
              all[g]=1
          val = [v for v in val if v not in all]
      if val:
        val.sort(key=lambda a: len(a))
        rel3[val[0]] = list(set(rel3.get(val[0],[])+val))

    #group by common prefix, infix, or suffix

    for key, val in rel3.items():
      val.sort(key=lambda a: len(a))
      len_val = len(val)
      for rng in range(0, len_val, 5):
          all = {}
          max_rng = min(rng+5, len_val)
          val2 = val[rng:max_rng]
          len_val2=len(val2)
          val2 = copy.deepcopy(val2)
          copy_val2 = copy.deepcopy(val2)
          for idx2, word in enumerate(copy_val2):
            if len(word) <= 4:
              continue
            for idx in range(idx2+1, len_val2):
              if type(val2[idx]) is tuple: 
                continue
              if type(val2[idx]) is str and (word in val2[idx] or val2[idx].startswith(word[:-1]) or val2[idx].startswith(word[:-2]) or val2[idx].startswith(word[:-3]) or val2[idx].endswith(word[1:]) or val2[idx].endswith(word[2:]) or val2[idx].endswith(word[2:])):
                val2[idx] = (word, val2[idx])
          val2 = [v for v in val2 if type(v) is tuple]
          itr = itertools.groupby(val2, lambda x : x[0])
          for key, group in itr:
            rel2[key] = list(set(rel2.get(key,[key]) + [v[1] for v in group]))
            all[key] = 1
            for v in group:
              all[v[1]] = 1
          #val3 = [v for v in val[rng:max_rng] if v not in all]
          #if val3:
          #  rel2[val3[0]] = val3

    len(rel), len(rel2), len(word2wncat)
    cat2word={}
    for key, value in word2wncat.items():
      cat2word[value]=cat2word.get(value,[])+[key]
    import json
    json.dump(rel2, open(f"{shared_dir}/conceptnet_ontology.json", "w", encoding="utf8"))
    json.dump(cat2word, open(f"{shared_dir}/conceptnet_ontology_cat2word.json", "w", encoding="utf8"))
    return rel2, cat2word
    
  #rel2 = create_rel()
  #list(rel2.items())

  def gather_ngram(self, arg, lang, force=True):
    shared_dir = arg.shared_dir
    ken_lm_location = arg.ken_lm_location
    if not force and path.exists(f"{shared_dir}/{lang}.arpa"):
      return
    if not os.path.exists("./lmplz"):
      os.system(f"cp {ken_lm_location} ./lmplz")
      os.system("chmod ugo+x ./lmplz")
    file = tokenize_data_subset(arg, lang, shared_dir, force=force)
    file2 = os.path.split(file)[-1]
    if not os.path.exists(file2) and not os.path.exists(file2.replace(".gz", "")):
      os.system(f"cp {file} ./{file2}")
    if os.path.exists(file2):
      os.system(f"gunzip ./{file2}")
    file2 = file2.replace(".gz", "")
    os.system(f"./lmplz --discount_fallback  --skip_symbols -o 5 --prune 5 --collapse_values  --arpa {lang}.arpa < ./{file2}")
    os.system(f"mv {lang}.arpa {shared_dir}")


  #### processing ngrams for various languages
  def load_ngrams(self, arg, arpafile, lang, ngram, compound_start, times=0, stopword_max_len=10, num_stopwords=75, force=False):
    shared_dir = arg.shared_dir
    non_words = "،♪↓↑→←━\₨₡€¥£¢¤™®©¶§←«»⊥∀⇒⇔√­­♣️♥️♠️♦️‘’¿*’-ツ¯‿─★┌┴└┐▒∎µ•●°。¦¬≥≤±≠¡×÷¨´:।`~�_“”/|!~@#$%^&*•()【】[]{}-_+–=<>·;…?:.,\'\"" 
    cjk = lang in ("ja", "zh", "ko")
    if cjk: stopword_max_len = 1

    if not force and os.path.exists(f"{shared_dir}/{lang}_ngram.tsv"):
      if not os.path.exis("./{lang}_ngram.tsv"):
        os.system("cp {shared_dir}/{lang}_ngram.tsv ./")
        os.system("cp {shared_dir}/{lang}_stopword.tsv ./")
      stopword_list = []
      with open(f"{lang}_stopword.tsv", "rb") as f:
        stopword_list=[l.split() for l in f.read().decode().split("\n")]
      stopword_list = [(l[1], float(l[0]) for l in stopword_list if len(l[1]) > 0 and len(l[1]) <= stopword_max_len]
      stopword_list.sort(key=lambda a: a[1], reverse=True)
      len_stopword_list = len(stopword_list)
      top_stopword = dict([item for item in stopword_list if len(item[0])<=2 and item[0][-1] in non_words] + \
                          [item for item in stopword_list[:min(len_stopword_list, num_stopwords)]])
      
      with open(f"{lang}_ngram.tsv", "rb",) as ngram_file:
          for line in ngram_file:
            line = line.decode().split()
            line = line[1:]
            hashKey = hash(tuple(line))
            ngram[hashKey] = weight
            compound = [line[0]]
            if not cjk:
              for l in line[1:]:
                if l.startswith(mt5_underscore): break 
                compound.append(l)
            compound=tuple(compound)
            compound_start[compound]=max(compound_start.get(compound,0), len(line))    

      return top_stopword, ngram, compound_start   

    file2 = os.path.split(arpafile)[-1]
    if not os.path.exists(file2):
      os.system(f"cp {arpafile} ./{file2}")
    stopword={}
    start = False
    t=0
    with open(f"_tmp_{lang}_ngram.tsv", "w", encoding="utf8") as ngram_file:
      with open(file2, "rb") as f:    
        for line in  f: #.read().decode().split("\n"): # do this incrementally if we need to. 
          line = line.decode().strip()
          if not line: continue
          if line.startswith("\\2-grams:"):
            start = True
          elif start:
            line = line.split()
            try:
              weight = float(line[0])
            except:
              continue
            weight = math.exp(float(weight))
            line = line[1:]
            if not line: continue
            if line[-1] == '0':
              line = line[:-1]
            if len(line) <= 1:
              continue
            if [l for l in line if l in non_words or l in ('<unk>', '<s>', '</s>')]: continue
            word = "".join(line)
            wordArr = word.split(mt5_underscore)
            if wordArr[0] == '':
              wordArr = wordArr[1:]
            if not cjk and not word.startswith(mt5_underscore): 
              continue
            if len(wordArr) > 1 and len(wordArr[0]) <= stopword_max_len:
              sw = wordArr[0].lower()
              stopword[sw] = stopword.get(sw,0) + weight               
            hashKey = hash(tuple(line))
            if hashKey not in ngram:
              ngram[hashKey] = weight
              ngram_file.write(str(weight)+"\t"+"\t".join(line)+"\n")

            #if len(wordArr) > 2:
            #  print (word, weight)
            #if t > 10000000:
            #  break
            t+=1
    top_stopword={} 
    print ('len stopword', len(stopword))
    if stopword:
      stopword_list = [l for l in stopword.items() if len(l[0]) > 0]
      stopword_list.sort(key=lambda a: a[1], reverse=True)
      len_stopword_list = len(stopword_list)
      top_stopword = dict([item for item in stopword_list if len(item[0])<=2 and item[0][-1] in non_words] + \
                          [item for item in stopword_list[:min(len_stopword_list, num_stopwords)]])
      # these are words that start sequences very often. 
      print ('top start word', len(top_stopword), top_stopword)

      with open (f"{lang}_stopword.tsv", "w", encoding="utf8") as o:
        for word, weight in stopword_list:
          o.write (str(weight)+"\t"+word+"\n")
      stopword=None
      stopword_list=None

    with open(f"{lang}_ngram.tsv", "w", encoding="utf8") as ngram_file2:
      with open(f"_tmp_{lang}_ngram.tsv", "rb") as ngram_file:
          for line in ngram_file:
            line = line.decode().split()
            weight = line[0]
            line = line[1:]
            weight = float(weight)
            wordArr = "".join(line).split(mt5_underscore)
            wordArr = wordArr[1:]
            if cjk and (line[0].lower() in top_stopword or line[-1].lower() in top_stopword):
              hashKey = hash(tuple(line))
              if hashKey in ngram:
                del ngram[hashKey]
              if line[0].lower() in top_stopword:
                line = line[1:]
              if line[-1].lower() in top_stopword:
                line = line[:-1]
              if len(line) > 1:
                hashKey = hash(tuple(line))
                if not hashKey in ngram:
                  ngram[hashKey] = weight
                  ngram_file2.write(str(weight)+"\t"+"\t".join(line)+"\n")
              else:
                continue
            elif len(wordArr) > 1 and (wordArr[0].lower() in top_stopword or wordArr[-1].lower() in top_stopword):
              hashKey = hash(tuple(line))
              if hashKey in ngram:
                del ngram[hashKey]
              new_line = []
              if wordArr[0].lower() in top_stopword:
                for l in line[1:]:
                  if l.startswith(mt5_underscore) or new_line:
                    new_line.append(l)
              else:
                new_line = line
              if new_line and wordArr[-1].lower() in top_stopword:
                new_line.reverse()
                i=-1
                for i, l in enumerate(new_line):
                  if l.startswith(mt5_underscore):
                    break
                new_line = new_line[i+1:]
                new_line.reverse()
              line = new_line
              if len(line) > 1:
                hashKey = hash(tuple(line))
                if not hashKey in ngram:
                  ngram[hashKey] = weight
                  ngram_file2.write(str(weight)+"\t"+"\t".join(line)+"\n")
              else:
                continue
            else:
              ngram_file2.write(str(weight)+"\t"+"\t".join(line)+"\n")
            compound = [line[0]]
            if not cjk:
              for l in line[1:]:
                if l.startswith(mt5_underscore): break 
                compound.append(l)
            compound=tuple(compound)
            compound_start[compound]=max(compound_start.get(compound,0), len(line))
      os.unlink(f"_tmp_{lang}_ngram.tsv")
      os.system("cp {lang}_ngram.tsv {shared_dir}")
      os.system("cp {lang}_stopword.tsv {shared_dir}")
      
    return top_stopword, ngram, compound_start   
  
###
# Future TODO:
#create a larger vocabulary for mt5 based on conceptnet and numberbatch. 
#create smaller embeddings size and factorization for the larger vocabulary to keep the modified mt5 model size reasonable
#store a large portion of the embeddings on disk via memmap so we can do out-of-core training and inference on the modified mt5
#incrementally increase the vocabulary and embeddings size by finding n-grams and fasttext training
  
  
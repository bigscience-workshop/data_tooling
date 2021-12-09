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

from datasets import load_dataset
import os
import re
import itertools
from re import finditer
import glob
import random
import fsspec
import json
from random import randint, choice
from collections import Counter
import spacy, itertools
import langid
from nltk.corpus import stopwords
import fsspec, os, gzip
from faker import Faker
from faker.providers import person, company, geo, address, ssn
from transformers import BertForTokenClassification, RobertaForTokenClassification, XLMRobertaForTokenClassification, M2M100ForConditionalGeneration, M2M100Tokenizer, MarianMTModel, AutoTokenizer, pipeline
import torch
import sys
from tqdm import tqdm

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                             os.path.pardir, os.path.pardir, os.path.pardir)))
import json
import data_tooling

from data_tooling.pii_processing.ontology.ontology_manager import OntologyManager

stopwords_en = set(stopwords.words('english'))
stopwords_wn = set(stopwords.words())

#junk_dict = dict([(a, 1) for a in "' 0123456789¯_§½¼¾×|†—~\"—±′–'°−{}[]·-\'?,./<>!@#^&*()+-‑=:;`→¶'"])

junk_dict = dict([(a, 1) for a in (
    " ~!@#$%^&*{}[]()_+=-0987654321`<>,./?':;“”\"\t\n\\πه☆●¦″"
    "．۩۱（☛₨➩°・■↑☻、๑º‹€σ٪’Ø·−♥ıॽ،٥《‘©。¨﴿！★×✱´٬→±x：¹？£―▷ф"
    "¡Г♫∟™ª₪®▬「—¯；¼❖․ø•�」٣，٢◦‑←§١ー٤）˚›٩▼٠«¢¸٨³½˜٭ˈ¿¬ι۞⌐¥►"
    "†ƒ∙²»¤…﴾⠀》′ا✓"
)])

rulebase = {"en": [([
      ("AGE", re.compile("\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old"), None, None, None),
      ("STREET_ADDRESS", re.compile(
          '\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)'), None, None, None),
      ("STREET_ADDRESS", re.compile('\b\d{5}(?:[-\s]\d{4})?\b'), None, None, None),
      ("STREET_ADDRESS", re.compile('P\.? ?O\.? Box \d+'), None, None, None),
      ("GOVT_ID", re.compile(
          '(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}'), None, None, None),
      ("DISEASE", re.compile("diabetes|cancer|HIV|AIDS|Alzheimer's|Alzheimer|heart disease"), None, None, None),
      ("NORP", re.compile("upper class|middle class|working class|lower class"), None, None, None),
      ], 1),
    ],
   "vi": []
  }
  
"""
Our langauges: 
- [ ] arabic
- [ ] bantu languages
- [ ] chinese
- [ ] french
- [ ] english
- [ ] indic languages
- [ ] portuguese
- [ ] spanish
- [ ] vietnamese
- [ ] basque
- [ ] catalan
- [ ] indonesian
"""
# note that we do not have a transformer model for catalan, but spacy covers catalan
lang2ner_model = {
      "sw": ("Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification ), # consider using one of the smaller models
      "yo": ("Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification ), 
      "ar": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "en": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "es": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "pt": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "fr": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "zh": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      'vi': ("lhkhiem28/COVID-19-Named-Entity-Recognition-for-Vietnamese", RobertaForTokenClassification),
      'hi': ("jplu/tf-xlm-r-ner-40-lang", XLMRobertaForTokenClassification ),
      'ur': ("jplu/tf-xlm-r-ner-40-lang", XLMRobertaForTokenClassification ),
      'id': ("jplu/tf-xlm-r-ner-40-lang", XLMRobertaForTokenClassification ), # also covers vietnamese
      'bn': ("sagorsarker/mbert-bengali-ner", BertForTokenClassification)
      }
  
"""
#sagorsarker/mbert-bengali-ner
Label ID	Label
0	O
1	B-PER
2	I-PER
3	B-ORG
4	I-ORG
5	B-LOC
6	I-LOC
"""



faker_map = dict([(a.split("_")[0], a) for a in [
    'ar_AA',
    'ar_PS',
    'ar_SA',
    'bg_BG',
    'cs_CZ',
    'de_AT',
    'de_CH',
    'de_DE',
    'dk_DK',
    'el_GR',
    'en_GB',
    'en_IE',
    'en_IN',
    'en_NZ',
    'en_TH',
    'en_US',
    'es_CA',
    'es_ES',
    'es_MX',
    'et_EE',
    'fa_IR',
    'fi_FI',
    'fr_CA',
    'fr_CH',
    'fr_FR',
    'fr_QC',
    'ga_IE',
    'he_IL',
    'hi_IN',
    'hr_HR',
    'hu_HU',
    'hy_AM',
    'id_ID',
    'it_IT',
    'ja_JP',
    'ka_GE',
    'ko_KR',
    'lt_LT',
    'lv_LV',
    'ne_NP',
    'nl_NL',
    'no_NO',
    'or_IN',
    'pl_PL',
    'pt_BR',
    'pt_PT',
    'ro_RO',
    'ru_RU',
    'sl_SI',
    'sv_SE',
    'ta_IN',
    'th_TH',
    'tr_TR',
    'tw_GH',
    'uk_UA',
    'zh_CN',
    'zh_TW']] + [('en', 'en_US')])

def remove_html_tags(text):
    """Remove html tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def stem(s):
    s = s.replace(".", " ").replace("!", " ").replace("?", " ").replace(",", " ").replace("-", " ").replace(";",
                                                                                                            " ").replace(
        "'", " '").replace("\"", " \"")
    sArr = s.lower().split()
    if len(sArr) > 4:
        sArr = sArr[:4]
    s = " ".join([s1[:4] if len(s1) > 4 else s1 for s1 in sArr if s1.strip()])
    return s


def pre_translation_steps(infile, target_lang='hi'):
    texts = []
    ner_mappings = []
    row_ids = []
    domains = []
    lbracket = "["
    rbracket = "]"
    if target_lang in ('zh', 'ja', 'ko'):
        lbracket = "[["
        rbracket = "]]"

    row_id = -1
    for s in tqdm(open(infile, "rb")):
        s = s.decode().strip()
        if not s: continue
        dat = json.loads(s)
        domain = dat.get('domain','')
        ner = dat.get('ner', {})
        text = dat['text']
        if 'id' not in dat:
            row_id += 1
        else:
            row_id = int(dat['id'])
        context = {}
        ner_mapping = {}
        seen = {}
        text = "... " + text + " "
        text = text.replace(lbracket, "(")
        text = text.replace(rbracket, ")", )
        if person_swap:
            _idx = 0
            for item in ner2:
                if item[0] in seen: continue
                seen[item[0]] = 1
                text = text.replace(item[0],
                                        ' ' + str(_idx) + " " + lbracket + item[0] + rbracket)
                ner_mapping[str(_idx) + " " + lbracket] = item
                _idx += 1

        texts.append(text)
        ner_mappings.append(ner_mapping)
        row_ids.append(row_id)
        domains.append(domain)
        return texts, ner_mappings, row_ids, domains


def chunks(lst, n):
    """Generate batches"""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def do_translations(texts, target_lang='hi', batch_size=16):
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = model.to('cuda').half()
    translations = []
    for src_text_list in tqdm(chunks(texts, batch_size)):
        batch = tokenizer(src_text_list, return_tensors="pt", padding=True, truncation=True).to('cuda')
        gen = model.generate(**batch, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
        outputs = tokenizer.batch_decode(gen, skip_special_tokens=True)
        translations.extend(outputs)
    return translations


def get_aligned_text(sent1, sent2, target_lang):
    """
    Given two sentences, find blocks of text that match and that don't match.
    return the blocks, and a matching score.
    Used to extract NER from original language sentence.
    TODO: prefer matching based on "[id] aaaa *" patterns
    """
    if target_lang in ("ja", "ko", "zh"):
      # splitting on spaces doesn't always work because some languages aren't space separated
      sep = ""
    else:
      sep = " "
      sent1 = sent1.split()
      sent2 = sent2.split()
    aMatch = difflib.SequenceMatcher(None,sent1, sent2)
    score = aMatch.ratio()
    blocks = aMatch.get_matching_blocks()
    blocks2 = []
    prevEndA = 0
    prevEndB = 0
    matchLen = 0
    nonMatchLen = 0
    print (blocks)
    for blockI in range(len(blocks)):
      if blockI > 0 or (blockI==0 and (blocks[blockI][0] != 0 or blocks[blockI][1] != 0)):
        blocks2.append([sep.join(sent1[prevEndA:blocks[blockI][0]]), sep.join(sent2[prevEndB:blocks[blockI][1]]), 0])
        nonMatchLen += max(blocks[blockI][0] - prevEndA, blocks[blockI][1] - prevEndB)
      if blocks[blockI][2] != 0:
        blocks2.append([sep.join(sent1[blocks[blockI][0]:blocks[blockI][0]+blocks[blockI][2]]), sep.join(sent2[blocks[blockI][1]:blocks[blockI][1]+blocks[blockI][2]]), 1])
        prevEndA = blocks[blockI][0]+blocks[blockI][2]
        prevEndB = blocks[blockI][1]+blocks[blockI][2]
        matchLen += blocks[blockI][2]
    #score = float(matchLen+1)/float(nonMatchLen+1)
    return (blocks2, score)

def post_translation_steps(outfile, translations, original_sentences, origional_ner,  ner_mappings, row_ids, domains, target_lang='hi'):
    rbracket = "]"
    if target_lang in ('zh', 'ja', 'ko'):
        rbracket = "]]"
    with open(outfile, "w", encoding="utf8") as o:
        for original_text, trans_text in zip(original_sentences, translations):
            index += 1
            ner_found = []
            trans_text = trans_text.lstrip(".")
            trans_text = trans_text.strip()
            if original_text: 
              trans_text = trans_text.replace(". [", " [").replace(".[", " [").replace("  ", " ")
              for key, ner_item in ner_mappings[index].items():
                if key in trans_text:
                    ner_found[key] = ner_item[1]
                elif key.replace(" ", "") in trans_text:
                    key = key.replace(" ", "")
                    ner_found[key] = ner_item[1]
                if target_lang in ('zh', 'ja', 'ko'):
                    trans_text.replace(" ", "")
                    trans_text.strip('#.')
            if ner_found:
                (blocks2, score) =  get_aligned_text(original_text, trans_text, target_lang)
                #since we know what slots we are replacing, 
                #we can align with the original sentence in the original language, 
                #and then extract the original NER value, or a translated version of the new value. 
                for (s1, s2, matched) in blocks2:   
                    s1 = s1.strip()
                    s2 = se2.strip() 
                    if "[" in s2 or "]" in s2:
                      key = s2.split("[")[1].split("]")[0]
                      if key in ner_found and key != s1:
                        ner_found[s1] = ner_found[key] 
                        del ner_found[key]
                for key in ner_found.keys():
                  if key not in oiriginal_text:
                    del ner_found[key]
                ner_found = list(ner_found.items())
                j = {'text': original_text, 'ner': ner_found, 'domain': domains[index],
                  'id': row_ids[index],
                  'lang': target_lang}
                o.write(json.dumps(j) + "\n")


#translate to en, do ner in en, do back-trans, match for ner mapping and map to original sentence, do ner in target lang.
def back_trans(target_lang, infile, outfile, rulebase):
  texts, ner_mappings, row_ids, domains = pre_translation_steps(infile, target_lang)
  en_text = do_translations(texts, 'en')
  with open("en_back_trans_"+infile, "w", encoding="utf8") as o:
    for text for en_text:
      o.write(json.dumps({"text": text})+"\n")
  apply_ner("en_back_trans_"+infile, "en_back_trans_"+outfile, rulebase.get('en', []), "en")
  en_text, ner_mappings2, _, _ = pre_translation_steps("en_back_trans_"+outfile, target_lang)
  back_trans_text = do_translations(en_text, target_lang)
  # TODO: resolve differences between ner_mappings2 and ner_mappings
  post_translation_steps("ner_1_"+outfile, back_trans_text, texts, ner_mapping, ner_mappings2, row_ids, domains, target_lang=target_lang)
  apply_ner("ner_1_"+outfile, outfile, rulebase.get(target_lang, []), target_lang)
  
def cjk_detect(texts):
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

def hf_ner(target_lang, hf_pipeline, text, predict_ner=None):
    """
    run the text through a Huggingface ner pipeline for the text. 
    any tags found by this method that contradicts what was already tagged will take precedence. 
    For example, if we previously had categorized a noun as an ORG (which spacy sometimes does), but 
    the HF ner pipeline categorized it as a PERSON, then the tag will be changed to PERSON.
    """
    is_cjk = target_lang in ("zh", "ja", "ko")
    if not isinstance(text, list):
      text_arr = [text]
    else:
      text_arr = text
    if isinstance(predict_ner, dict):
      predict_ner = [predict_ner]
    if predict_ner is None:
      predict_ner = [{} for i in range(len(text_arr))]
    results_arr = hf_pipeline(text_arr)
    for text, results, ner in zip(text_arr, results_arr, predict_ner):
      print (results)
      results.sort(key=lambda a: a['start'])
      prev_word = [0,0]
      prev_label = None
      prev_start = None
      start = 0
      for ner_result in results:
        print (ner_result)
        if True:
            start = ner_result['start']
            is_cjk = cjk_detect(text[ner_result['start']:ner_result['end']])
            #print (text[ner_result['start']:ner_result['end']], is_cjk)
            if not is_cjk:
              if text[start] not in " {}[]()\"'“”《 》« »":
                for j in range(1, start):
                  if start - j == -1 or text[start-j] in " {}[]()\"'“”《 》« »":
                    start = max(start -j, 0)
                    break
              word = text[start:].strip().split(" ",1)[0]
              if len(word) < ner_result['end'] - start:
                end = ner_result['end']
              else:
                end = start+len(word)
            else:
              start = ner_result['start']
              end = ner_result['end']
        print (start, end)
        if prev_label is not None and (ner_result['entity'].startswith('B-') or prev_label != ner_result['entity'].split("-")[-1].upper() or\
                                       (prev_word[-1] < start and prev_word[0] != prev_word[1])):
            if ner_result['entity'].startswith('B-') and prev_word[1] > start:
              prev_word[1] = start
            ner_word = text[prev_word[0]:prev_word[1]].strip(" {}[]()\"'“”《 》« »")
            print ('**', ner_word, prev_label, prev_word, start)
            if ner_word not in ner or prev_label not in ner[ner_word]:
              if ner_word:
                if prev_label == 'PER': prev_label = 'PERSON'
                #remove overlaps
                for key in list(ner.keys()):
                  if " "+key in ner_word or key+" " in ner_word:# or " "+ent in key or ent+" " in key:
                    del ner[key]
                  elif len(key) > 6 and key in ner_word: # parameterize 6
                    del ner[key]
                ner[ner_word] = prev_label
              prev_label = ner_result['entity'].split("-")[-1].upper() 
              prev_word = [start, end]
        elif prev_label is None:
          prev_label = ner_result['entity'].split("-")[-1].upper() 
          prev_word = [start, end]
        else:  
          prev_word[-1] = max(prev_word[-1], end)
      if prev_word[0] != prev_word[1]:
          ner_word = text[prev_word[0]:prev_word[1]].strip(" {}[]()\"'“”《 》« »")
          if ner_word not in ner or prev_label not in ner[ner_word]:
            if ner_word:
              if prev_label == 'PER': prev_label = 'PERSON'
              #remove overlaps
              for key in list(ner.keys()):
                if " "+key in ner_word or key+" " in ner_word:# or " "+ent in key or ent+" " in key:
                  del ner[key]
                elif len(key) > 6 and key in ner_word: # parameterize 6
                  del ner[key]
              ner[ner_word] = prev_label
                      
    return predict_ner

from data_tooling.pii_processing.ontology.stopwords import stopwords as stopwords_ac_dc



# do first a lexicon match, than a spacy match if any, and then do progressive groups of regex, then do hf_tranformers. 
#TODO - do complicated rules, such as PERSON Inc. => ORG
#TODO - check for zh working properly
def apply_ner(infile, outfile, rule_base, target_lang, do_ontology_manager=True, do_spacy_if_avail=True, do_hf=True, char_before_after_window=10):
  stopwords_target_lang = set(stopwords_ac_dc[target_lang])
  nlp = None
  if do_spacy_if_avail:
    if target_lang == 'en':
      nlp = spacy.load('en_core_web_sm')
    elif target_lang == 'zh':
      nlp = spacy.load('zh_core_web_sm')
    elif target_lang == 'pt':
      nlp = spacy.load('pt_core_news_sm')
    elif target_lang == 'fr':
      nlp = spacy.load('fr_core_news_sm')
    elif target_lang == 'ca':
      nlp = spacy.load('ca_core_news_sm')
  model = None
  hf_pipeline = None
  if do_hf:
    model_name, model_cls = lang2ner_model.get(target_lang, [None, None])
    if model_cls is not None:
      model = model_cls.from_pretrained(model_name)
      hf_pipeline = pipeline("ner", model=model, tokenizer=AutoTokenizer.from_pretrained(model_name))
  if do_ontology_manager:
    ontology_manager = OntologyManager(target_lang=target_lang)
  else:
    ontology_manager = None
  right = {}
  wrong = {}
  with open(outfile, "w", encoding="utf8") as o:
    for line in tqdm(open(infile, "rb")):
        pred = [] #predicted regex rules ent:label
        recs = json.loads(line)
        if not isinstance(recs, list):
          recs = [recs]
        for d in recs:
          print (d)
          text = d['text']
          if not text: continue
          predict_ner = d.get('ner',{})
          if ontology_manager:
            parsed = ontology_manager.tokenize(text)
            cunk2ner = list(parsed['chunk2ner'].items())
            for c in cunk2ner:
              key, val = c[0][0].replace(" ", "").replace("_", "").replace("_", "") if target_lang in  ('zh', 'ja', 'ko') else c[0][0].replace("_", " ").replace("_", " "), c[1]
              predict_ner[key] = val
          if nlp:
            doc = nlp(text)
            entities = list(doc.ents)
            ents = dict(list(set([(entity.text, entity.label_) for entity in entities if
              entity.label_ in ('PERSON', 'GPE', 'ORG', 'NORP') and 'http:' not in entity.text])))
            for ent, label in ents.items():
              if ent in predict_ner and ('PERSON' if predict_ner[ent] == 'PUBLIC_FIGURE' else predict_ner[ent]) ==label:
                #print ("matched")
                continue
              else:
                #remove simple overlap
                for key in list(predict_ner.keys()):
                  if " "+key in ent or key+" " in ent: # or " "+ent in key or ent+" " in key:
                    if predict_ner[key] == 'PUBLIC_FIGURE':
                      label = "PUBLIC_FIGURE"
                    del predict_ner[key]
                predict_ner[ent] = label
          rule_level = -1
          ner = d['ner'] # the human tagged data
          for rule_groups, ntimes in rule_base:
            rule_level += 1
            for times in range(ntimes):
              rule_id = -1
              for label, regex, old_label, before, after in rule_groups:
                rule_id += 1
                if old_label is None and times > 0: continue  
                for ent in regex.findall(text):
                  if type(ent) != str: continue
                  if old_label is not None and (ent not in predict_ner or predict_ner[ent] != old_label): continue
                  t = text
                  len_ent = len(ent)
                  while ent in t:
                    i = t.index(ent)
                    if before:
                      before_t[max(0, i-char_before_after_window): i]
                      if before not in before_t:
                        t = t[i + len_ent:]
                        continue
                    if after:
                      after_t[i+len_x: min(len_text, i+len_ent+char_before_after_window)]
                      if after not in after_t:
                        t = t[i + len_ent:]
                        continue
                    #remove simple overlap
                    for key in list(predict_ner.keys()):
                      if " "+key in ent or key+" " in ent:# or " "+ent in key or ent+" " in key:
                        del predict_ner[key]
                    pred.append((ent, label, rule_id, rule_level))
                    predict_ner[ent] = label
                    break
          if hf_pipeline is not None:
            hf_ner(target_lang, hf_pipeline, text, predict_ner)
          d['ner'] = list(predict_ner.items())
          o.write(json.dumps(d)+"\n")



if __name__ == "__main__":

    initial = target_lang = None
    if "-initial" in sys.argv:
      initial = sys.argv[sys.argv.index("-initial")+1]   
    if "-target_lang" in sys.argv:
      target_lang = sys.argv[sys.argv.index("-target_lang")+1]   
    if target_lang:
      #TODO - load the rulebase dynamically from pii_processing.regex folder a file of the form <initial>_<target_lang>.py
      infile = f"{target_lang}.jsonl"
      outfile = "predicted_"+infile
      back_trans(target_lang, infile, outfile, rulebase)
      


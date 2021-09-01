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
from datasets import load_dataset
from collections import Counter
from itertools import chain
import os
import re
import glob
import math
from faker import Faker
from faker.providers import person, internet, geo, address, date_time, job, bank, credit_card, ssn
from nltk.corpus import stopwords
import difflib
import random
import torch
import nltk
from nltk.corpus import stopwords
import base64, hashlib
from random import choice
import spacy, neuralcoref, itertools
from collections import Counter, OrderedDict
trannum = str.maketrans("0123456789", "1111111111")
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir, os.path.pardir)))

from data_tooling.processors.processor import Processor
from data_tooling.translation.backtrans import BackTranslate

class PIIProcessor (Processor):
  """
    A Personally Identifiable Information (PII) and Name Entity Recognition (NER) manager. 
    Operates in several supported languages.

    Will parse a text and create a template for a sentence, with the slots filled in with masked ids.
      - John went to the store, and he bought gum. => [1] went to the store, and he bought gum.

    Can be used to anonymize and do gender swapping:
      - John went to the store, and he bought gum. => Jane went to the store, and she bought gum.

    Can extract various PII/NER categories described below.

    Can create a hash code for certain categories of PII.

    In addition to Spacy's NER:

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

    PII/NER information extracted from a sentence also include:

      PRONOUN
      GENDER
      TITLE
      RELIGION
      RACE # TODO
      JOB
      POLITICAL_PARTY #TODO
      UNION_MEMBERSHIP #TODO
      DISEASE
      DOMAIN_NAME
      EMAIL_ADDRESS
      IP_ADDRESS
      PRICE
      CREDIT_CARD
      CRYPTO
      STREET_ADDRESS
      US_SSN

  """

  # class varaiables

  pii_ontology = {
      'PRONOUN': ['PRONOUN', 'PERSON'],
      'GENDER': ['GENDER', 'PERSON'],
      'TITLE': ['TITLE', 'PERSON'],
      'JOB': ['JOB', 'PERSON'],
      'RELIGION': ['RELIGION', 'NORP'],
      'RACE': ['RACE', 'NORP'],
      'DISEASE': ['DISEASE'],
      'POLITICAL_PARTY': ['POLITICAL_PARTY', 'ORG'],
      'UNION_MEMBERSHIP': ['UNION_MEMBERSHIP', 'ORG'],
      'STREET_ADDRESS': ['STREET_ADDRESS', 'PLACE'],
      'CRYPTO': ['CRYPTO'],
      'CREDIT_CARD': ['CREDIT_CARD'],
      'PRICE': ['PRICE', 'MONEY'],
      'DOMAIN_NAME': ['DOMAIN_NAME', 'ELECTRONIC_ADDRESS'],
      'EMAIL_ADDRESS': ['EMAIL_ADDRESS', 'ELECTRONIC_ADDRESS'],
      'IP_ADDRESS': ['IP_ADDRESS', 'ELECTRONIC_ADDRESS'],
      'US_SSN': ['US_SSN', 'CARDINAL'],
  }

  faker_en = None
  faker_map = {
      'es': 'es_ES',
      'en': 'en_US',
      'ar': 'ar_AA',
      'pt': 'pt_PT',
      'fr': 'fr_FR',
      'hi': 'hi_IN',
      'zh': 'zh_CN',
      # TODO, do the rest of the faker supported languages
    }
 
  #from https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py which is under the MIT License
  date_regex             = re.compile('(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}', re.IGNORECASE)
  time_regex             = re.compile('\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?', re.IGNORECASE)
  phone_regex            = re.compile('''((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))''')
  phones_with_exts_regex = re.compile('((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))', re.IGNORECASE)
  link_regex             = re.compile('(?i)((?:https?://|www\d{0,3}[.])?[a-z0-9.\-]+[.](?:(?:international)|(?:construction)|(?:contractors)|(?:enterprises)|(?:photography)|(?:immobilien)|(?:management)|(?:technology)|(?:directory)|(?:education)|(?:equipment)|(?:institute)|(?:marketing)|(?:solutions)|(?:builders)|(?:clothing)|(?:computer)|(?:democrat)|(?:diamonds)|(?:graphics)|(?:holdings)|(?:lighting)|(?:plumbing)|(?:training)|(?:ventures)|(?:academy)|(?:careers)|(?:company)|(?:domains)|(?:florist)|(?:gallery)|(?:guitars)|(?:holiday)|(?:kitchen)|(?:recipes)|(?:shiksha)|(?:singles)|(?:support)|(?:systems)|(?:agency)|(?:berlin)|(?:camera)|(?:center)|(?:coffee)|(?:estate)|(?:kaufen)|(?:luxury)|(?:monash)|(?:museum)|(?:photos)|(?:repair)|(?:social)|(?:tattoo)|(?:travel)|(?:viajes)|(?:voyage)|(?:build)|(?:cheap)|(?:codes)|(?:dance)|(?:email)|(?:glass)|(?:house)|(?:ninja)|(?:photo)|(?:shoes)|(?:solar)|(?:today)|(?:aero)|(?:arpa)|(?:asia)|(?:bike)|(?:buzz)|(?:camp)|(?:club)|(?:coop)|(?:farm)|(?:gift)|(?:guru)|(?:info)|(?:jobs)|(?:kiwi)|(?:land)|(?:limo)|(?:link)|(?:menu)|(?:mobi)|(?:moda)|(?:name)|(?:pics)|(?:pink)|(?:post)|(?:rich)|(?:ruhr)|(?:sexy)|(?:tips)|(?:wang)|(?:wien)|(?:zone)|(?:biz)|(?:cab)|(?:cat)|(?:ceo)|(?:com)|(?:edu)|(?:gov)|(?:int)|(?:mil)|(?:net)|(?:onl)|(?:org)|(?:pro)|(?:red)|(?:tel)|(?:uno)|(?:xxx)|(?:ac)|(?:ad)|(?:ae)|(?:af)|(?:ag)|(?:ai)|(?:al)|(?:am)|(?:an)|(?:ao)|(?:aq)|(?:ar)|(?:as)|(?:at)|(?:au)|(?:aw)|(?:ax)|(?:az)|(?:ba)|(?:bb)|(?:bd)|(?:be)|(?:bf)|(?:bg)|(?:bh)|(?:bi)|(?:bj)|(?:bm)|(?:bn)|(?:bo)|(?:br)|(?:bs)|(?:bt)|(?:bv)|(?:bw)|(?:by)|(?:bz)|(?:ca)|(?:cc)|(?:cd)|(?:cf)|(?:cg)|(?:ch)|(?:ci)|(?:ck)|(?:cl)|(?:cm)|(?:cn)|(?:co)|(?:cr)|(?:cu)|(?:cv)|(?:cw)|(?:cx)|(?:cy)|(?:cz)|(?:de)|(?:dj)|(?:dk)|(?:dm)|(?:do)|(?:dz)|(?:ec)|(?:ee)|(?:eg)|(?:er)|(?:es)|(?:et)|(?:eu)|(?:fi)|(?:fj)|(?:fk)|(?:fm)|(?:fo)|(?:fr)|(?:ga)|(?:gb)|(?:gd)|(?:ge)|(?:gf)|(?:gg)|(?:gh)|(?:gi)|(?:gl)|(?:gm)|(?:gn)|(?:gp)|(?:gq)|(?:gr)|(?:gs)|(?:gt)|(?:gu)|(?:gw)|(?:gy)|(?:hk)|(?:hm)|(?:hn)|(?:hr)|(?:ht)|(?:hu)|(?:id)|(?:ie)|(?:il)|(?:im)|(?:in)|(?:io)|(?:iq)|(?:ir)|(?:is)|(?:it)|(?:je)|(?:jm)|(?:jo)|(?:jp)|(?:ke)|(?:kg)|(?:kh)|(?:ki)|(?:km)|(?:kn)|(?:kp)|(?:kr)|(?:kw)|(?:ky)|(?:kz)|(?:la)|(?:lb)|(?:lc)|(?:li)|(?:lk)|(?:lr)|(?:ls)|(?:lt)|(?:lu)|(?:lv)|(?:ly)|(?:ma)|(?:mc)|(?:md)|(?:me)|(?:mg)|(?:mh)|(?:mk)|(?:ml)|(?:mm)|(?:mn)|(?:mo)|(?:mp)|(?:mq)|(?:mr)|(?:ms)|(?:mt)|(?:mu)|(?:mv)|(?:mw)|(?:mx)|(?:my)|(?:mz)|(?:na)|(?:nc)|(?:ne)|(?:nf)|(?:ng)|(?:ni)|(?:nl)|(?:no)|(?:np)|(?:nr)|(?:nu)|(?:nz)|(?:om)|(?:pa)|(?:pe)|(?:pf)|(?:pg)|(?:ph)|(?:pk)|(?:pl)|(?:pm)|(?:pn)|(?:pr)|(?:ps)|(?:pt)|(?:pw)|(?:py)|(?:qa)|(?:re)|(?:ro)|(?:rs)|(?:ru)|(?:rw)|(?:sa)|(?:sb)|(?:sc)|(?:sd)|(?:se)|(?:sg)|(?:sh)|(?:si)|(?:sj)|(?:sk)|(?:sl)|(?:sm)|(?:sn)|(?:so)|(?:sr)|(?:st)|(?:su)|(?:sv)|(?:sx)|(?:sy)|(?:sz)|(?:tc)|(?:td)|(?:tf)|(?:tg)|(?:th)|(?:tj)|(?:tk)|(?:tl)|(?:tm)|(?:tn)|(?:to)|(?:tp)|(?:tr)|(?:tt)|(?:tv)|(?:tw)|(?:tz)|(?:ua)|(?:ug)|(?:uk)|(?:us)|(?:uy)|(?:uz)|(?:va)|(?:vc)|(?:ve)|(?:vg)|(?:vi)|(?:vn)|(?:vu)|(?:wf)|(?:ws)|(?:ye)|(?:yt)|(?:za)|(?:zm)|(?:zw))(?:/[^\s()<>]+[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019])?)', re.IGNORECASE)
  email_regex            = re.compile("([a-z0-9!#$%&'*+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)", re.IGNORECASE)
  ip_regex               = re.compile('(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', re.IGNORECASE)
  ipv6_regex             = re.compile('\s*(?!.*::.*::)(?:(?!:)|:(?=:))(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)){6}(?:[0-9a-f]{0,4}(?:(?<=::)|(?<!::):)[0-9a-f]{0,4}(?:(?<=::)|(?<!:)|(?<=:)(?<!::):)|(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-4]|2[0-4]\d|1\d\d|[1-9]?\d)){3})\s*', re.VERBOSE|re.IGNORECASE|re.DOTALL)
  price_regex            = re.compile('[$]\s?[+-]?[0-9]{1,3}(?:(?:,?[0-9]{3}))*(?:\.[0-9]{1,2})?')
  credit_card_regex      = re.compile('((?:(?:\\d{4}[- ]?){3}\\d{4}|\\d{15,16}))(?![\\d])')
  btc_address_regex      = re.compile('(?<![a-km-zA-HJ-NP-Z0-9])[13][a-km-zA-HJ-NP-Z0-9]{26,33}(?![a-km-zA-HJ-NP-Z0-9])')
  street_address_regex   = re.compile('\d{1,4} [\w\s]{1,20}(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)', re.IGNORECASE)
  zip_code_regex         = re.compile(r'\b\d{5}(?:[-\s]\d{4})?\b')
  po_box_regex           = re.compile(r'P\.? ?O\.? Box \d+', re.IGNORECASE)
  ssn_regex              = re.compile('(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}')
  # see also https://stackoverflow.com/questions/5590862/icd9-regex-pattern - but this could be totally wrong!
  icd_code_regex         = re.compile('[A-TV-Z][0-9][A-Z0-9](\.[A-Z0-9]{1,4})', re.IGNORECASE)

  # we do regex in this order in order to not capture ner inside domain names and email addresses.
  default_ner_regexes = OrderedDict([
    ("DOMAIN_NAME"     , link_regex),                                  
    ("EMAIL_ADDRESS"   , email_regex),
    ("DATE"             ,date_regex),
    ("TIME"            , time_regex),
    ("PHONE_NUMBER"    , [phone_regex, phones_with_exts_regex]),
    ("IP_ADDRESS"      , [ip_regex, ipv6_regex,]),
    ("PRICE"           , price_regex),
    ("CREDIT_CARD"     , credit_card_regex),
    ("CRYPTO"          , btc_address_regex),
    ("STREET_ADDRESS"  , [street_address_regex, zip_code_regex, po_box_regex]),
    ("US_SSN"          , ssn_regex),
    ("ICD_CODE"        , icd_code_regex)
  ])

  pronoun = {
            "he": "he",  
            "He": "he",  
            "She": "she",
            "She": "she",  
            "his": "he",  
            "His": "he",  
            "him": "he",  
            "Him": "he",  
            "Her": "she",
            "her": "she",  
            "Hers": "she",
            "hers": "she", 
  }

  title2pronoun = {
            "Mr.": "he",   
            "Mr": "he",
            "Ms": "she",
            "Ms.": "she",
            "Mrs": "she",
            "Mrs.": "she",
            "Miss": "she",
            "Miss.": "she",
            "mr.": "he",   
            "mr": "he",
            "ms": "she",
            "ms.": "she",
            "mrs": "she",
            "mrs.": "she",
            "miss": "she",
            "miss.": "she",
  }

  pronoun2itle = {
      "he": ["Mr.", "Mr",],
      "she": ["Ms.", "Mrs.", "Miss.", "Ms", "Mrs", "Miss"],
      "they": ["Mr.", "Ms."]
  }

  #choose a smaller list, because the bigger list of genders, such as the one from Facebook, does not work with with back translations.
  #TODO: get gender terms in target_lang. See https://github.com/christabor/faker_extras/blob/master/faker_extras/human.py
  gender2pronoun = {
            'Nonbinary': "they",
            'Man': "he",
            'Woman': "she",
            'Boy': "he",
            'Girl': "she",
            'Female' : "she",
            'Male' : "she",
            'Lesbian': "she",
            'Gay Man': "he",
            'Gay Woman': "he",
            'Transgender Man': "he",
            'Transgender Woman': "she",

            'Young Nonbinary': "they",
            'Young Man': "he",
            'Young Woman': "she",
            'Young Boy': "he",
            'Young Girl': "she",
            'Young Female' : "she",
            'Young Male' : "she",
            'Young Lesbian': "she",
            'Young Gay Man': "he",
            'Young Gay Woman': "he",
            'Young Transgender Man': "he",
            'Young Transgender Woman': "she",

            'Adult Nonbinary': "they",
            'Adult Man': "he",
            'Adult Woman': "she",
            'Adult Female' : "she",
            'Adult Male' : "she",
            'Adult Lesbian': "she",
            'Adult Gay Man': "he",
            'Adult Gay Woman': "he",
            'Adult Transgender Man': "he",
            'Adult Transgender Woman': "she",

            'Older Nonbinary': "they",
            'Older Man': "he",
            'Older Woman': "she",
            'Older Female' : "she",
            'Older Male' : "she",
            'Older Lesbian': "she",
            'Older Gay Man': "he",
            'Older Gay Woman': "he",
            'Older Transgender Man': "he",
            'Older Transgender Woman': "she",
                        
        }
  
  #TODO, add in age range
  pronoun2gender = {
      "they": ["Nonbinary"],
      "he": ["Man", "Boy", "Male", "Gay Man", "Transgender Man"],
      "she": ["Woman", "Girl", "Female", "Lesbian", "Gay Woman", "Transgender Woman"]
  }

  #adapted from https://github.com/christabor/faker_extras/blob/master/faker_extras/human.py, licensed under MIT License
  person2religion = {
            'Atheist': 'Atheism',
            'Christian': 'Christianity',
            'Muslim': 'Islam',
            'Hindu': 'Hinduism',
            'Buddhist': 'Buddhism',
            'Sikh': 'Sikhism',
            'Jew': 'Judaism',
            "Bahá'í": 'Bahaism',
            'Confucianists': 'Confucianism',
            'Jain': 'Jainism',
            'Shintoists': 'Shintoism',
  }

  pronoun_swap = {'he': 'she', 'He': 'She', 'his': 'her', 'His': 'Her', \
                    'she': 'he', 'She': 'He', 'her': 'his', 'hers': 'his', 'Her': 'His', 'Hers': 'His', }
  
  title_swap = {"mr.": "Mrs.", "mrs.": "Mr.", "miss.": "Mr.", "mr": "Mrs", "mrs": "Mr", "miss": "Mr"}

  # from https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/generalized_swaps.txt and
  # from https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/extra_gendered_words.txt which are licensed under MIT license
  gender_swap = {
      'actor':'actress', 
      'actors':'actresses',
      'actress':'actor', 
      'actresses':'actors',
      'airman':'airwoman', 
      'airmen':'airwomen', 
      'airwoman':'airman', 
      'airwomen':'airmen', 
      'aunt':'uncle', 
      'aunts':'uncles',
      'boy':'girl', 
      'boys':'girls', 
      'bride':'groom',
      'brides':'grooms', 
      'brother':'sister', 
      'brothers':'sisters', 
      'businessman':'businesswoman', 
      'businessmen':'businesswomen',
      'businesswoman':'businessman', 
      'businesswomen':'businessmen',
      'chairman':'chairwoman',
      'chairmen':'chairwomen',
      'chairwoman':'chairman', 
      'chairwomen':'chairman',
      'chick':'dude', 
      'chicks':'dudes', 
      'dad':'mom', 
      'dads':'moms',
      'daddy':'mommy',
      'daddies':'mommies', 
      'daughter':'son',
      'daughters':'sons', 
      'dude':'chick', 
      'dudes':'chicks', 
      'father':'mother', 
      'fathers':'mothers', 
      'female':'male',
      'females':'males', 
      'gal':'guy', 
      'gals':'guys', 
      'gentleman':'lady',
      'gentlemen':'ladies',
      'girl':'boy', 
      'girls':'boys', 
      'granddaughter':'grandson',
      'granddaughters':'grandsons', 
      'grandson':'granddaughter', 
      'grandsons':'granddaughters',
      'groom':'bride', 
      'grooms':'brides',
      'guy':'girl', 
      'guys':'girls', 
      'he':'she', 
      'herself':'himself', 
      'him':'her', 
      'himself':'herself', 
      'his':'her', 
      'husband':'wife',
      'husbands':'wives', 
      'king':'queen', 
      'kings':'queens',
      'ladies':'gentlemen',
      'lady':'gentleman', 
      'lord':'lady',
      'lords':'ladies', 
      'ma\'am':'sir', 
      'male':'female',
      'males':'females', 
      'man':'woman', 
      'men':'women', 
      'miss':'sir', 
      'mom':'dad', 
      'moms':'dads',
      'mommy':'daddy', 
      'mommies':'daddies',
      'mother':'father', 
      'mothers':'fathers', 
      'mr.':'mrs.', 
      'mrs.':'mr.', 
      'ms.':'mr.', 
      'policeman':'policewoman',
      'policewoman':'policeman', 
      'prince':'princess', 
      'princes':'princesses',
      'princess':'prince', 
      'princesses':'princes',
      'queen':'king', 
      'queens':'kings',
      'she':'he', 
      'sir':'ma''am', 
      'sister':'brother', 
      'sisters':'brothers', 
      'son':'daughter', 
      'sons':'daughters', 
      'spokesman':'spokeswoman',
      'spokesmen':'spokeswomen', 
      'spokeswoman':'spokesman', 
      'spokeswomen':'spokesmen',
      'uncle':'aunt', 
      'uncles':'aunts',
      'wife':'husband', 
      'wives':'husbands',
      'woman':'man', 
      'women':'men', 
      'cowboy':'cowgirl',
      'cowboys':'cowgirls',
      'camerawomen':'cameramen',
      'cameraman':'camerawoman',
      'busboy':'busgirl',
      'busboys':'busgirls',
      'bellboy':'bellgirl',
      'bellboys':'bellgirls',
      'barman':'barwoman',
      'barmen':'barwomen',
      'tailor':'seamstress',
      'tailors':'seamstress',
      'prince':'princess',
      'princes':'princesses',
      'governor':'governess',
      'governors':'governesses',
      'adultor':'adultress',
      'adultors':'adultresses',
      'god':'godess',
      'gods':'godesses',
      'host':'hostess',
      'hosts':'hostesses',
      'abbot':'abbess',
      'abbots':'abbesses',
      'actor':'actress',
      'actors':'actresses',
      'bachelor':'spinster',
      'bachelors':'spinsters',
      'baron':'baroness',
      'barons':'barnoesses',
      'beau':'belle',
      'beaus':'belles',
      'bridegroom':'bride',
      'bridegrooms':'brides',
      'brother':'sister',
      'brothers':'sisters',
      'duke':'duchess',
      'dukes':'duchesses',
      'emperor':'empress',
      'emperors':'empresses',
      'enchanter':'enchantress',
      'father':'mother',
      'fathers':'mothers',
      'fiance':'fiancee',
      'fiances':'fiancees',
      'priest':'nun',
      'priests':'nuns',
      'gentleman':'lady',
      'gentlemen':'ladies',
      'grandfather':'grandmother',
      'grandfathers':'grandmothers',
      'headmaster':'headmistress',
      'headmasters':'headmistresses',
      'hero':'heroine',
      'heros':'heroines',
      'lad':'lass',
      'lads':'lasses',
      'landlord':'landlady',
      'landlords':'landladies',
      'male':'female',
      'males':'females',
      'man':'woman',
      'men':'women',
      'manservant':'maidservant',
      'manservants':'maidservants',
      'marquis':'marchioness',
      'masseur':'masseuse',
      'masseurs':'masseuses',
      'master':'mistress',
      'masters':'mistresses',
      'monk':'nun',
      'monks':'nuns',
      'nephew':'niece',
      'nephews':'nieces',
      'priest':'priestess',
      'priests':'priestesses',
      'sorcerer':'sorceress',
      'sorcerers':'sorceresses',
      'stepfather':'stepmother',
      'stepfathers':'stepmothers',
      'stepson':'stepdaughter',
      'stepsons':'stepdaughters',
      'steward':'stewardess',
      'stewards':'stewardesses',
      'uncle':'aunt',
      'uncles':'aunts',
      'waiter':'waitress',
      'waiters':'waitresses',
      'widower':'widow',
      'widowers':'widows',
      'wizard':'witch',
      'wizards':'witches'
  }

  # from https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/data/female_occupations.txt which is under MIT license
  # use for adding into all jobs list. consider what else we can do with this list.
  female_jobs = [
      'attendant',
      'cashier',
      'teacher',
      'nurse',
      'assistant',
      'secretary',
      'auditor',
      'cleaner',
      'receptionist',
      'clerk',
      'counselor',
      'designer',
      'hairdresser',
      'writer',
      'housekeeper',
      'baker',
      'accountant',
      'editor',
      'librarian',
      'tailor',                
  ]
  # from https://github.com/uclanlp/corefBias/blob/master/WinoBias/wino/data/male_occupations.txt which is under MIT license
  # use for adding into all jobs list. consider what else we can do with this list.
  male_jobs = [
      'driver',
      'supervisor',
      'janitor',
      'cook',
      'mover',
      'laborer',
      'construction worker',
      'chief',
      'developer',
      'carpenter',
      'manager',
      'lawyer',
      'farmer',
      'salesperson',
      'physician',
      'guard',
      'analyst',
      'mechanic',
      'sheriff',
      'CEO'
  ]

  # for race proxies, we could also include physical descriptions, such as skin color and hair style. social economic status? debt level and income level? TODO.
  # see https://engineering.cmu.edu/news-events/news/2018/12/11-datta-proxies.html
  race_list = []

  #adapted from https://github.com/christabor/faker_extras/blob/master/faker_extras/human.py, licensed under MIT License
  race_list0 = [
            'Aboriginal',
            'Australian',
            'South Pacific',
            'Aborigine',
            'African',
            'African-American',
            'American',
            'American Indian',
            'Arabian',
            'Arabic',
            'Arab',
            'Middle Eastern',
            'Asian',
            'Asian-American',
            'Asian Indian',
            'Asian Mongoloid',
            'Asian Subcontinent',
            'Asian Pacific',
            'Bi-multiracial',
            'Black',
            'African descent',
            'Black',
            'African-American',
            'Central-Southern African',
            'Brown',
            'Hispanic',
            'Chinese',
            'Eastern Indian',
            'Eskimo',
            'Aleutian',
            'European',
            'Filipino',
            'Hispanic',
            'Indian',
            'Middle Asian',
            'Pakistani',
            'Islander',
            'Japanese',
            'Jewish',
            'Korean',
            'Latina',
            'Latino',
            'Mestiza',
            #'Mixed',
            'Mexican',
            'Middle Eastern',
            'Native American',
            'Aborigine',
            'Indigenous People',
            'Pacific Islander',
            'East Asian',
            'Polynesian',
            'Pacific Islander',
            'South American',
            'Latin American',
            'Vietnamese',
            'White',
            'Caucasian',
            'European',
            'Northern European',
        ]

  #from https://www.cdc.gov/diseasesconditions/az/z.html
  disease_list = [
            'ADHD',
            'Arthritis',
            'Asthma',
            'Autism',
            'Avian Influenza',
            'Birth Defects',
            'Cancer',
            'Chlamydia',
            'Chronic Fatigue Syndrome',
            'Chronic Obstructive Pulmonary Disease',
            'COPD',
            'COVID-19',
            'Diabetes',
            'Ebola',
            'Ebola Virus Disease',
            'Epilepsy',
            'Fetal Alcohol Spectrum Disorder',
            'Flu',
            'Influenza',
            'Genital Herpes',
            'Herpes Simplex Virus',
            'Giardiasis',
            'Gonorrhea',
            #'Healthy Water # not sure if this is a disease or health condition
            'Heart Disease',
            'Hepatitis',
            'HIV/AIDS',
            'Human papillomavirus',
            'HPV',
            'Kidney Disease',
            'Chronic Kidney Disease',
            'Meningitis',
            'Methicillin-resistant Staphylococcus aureus',
            'MRSA',
            'Microcephaly',
            'Middle East Respiratory Syndrome',
            'MERS',
            'Overweight',
            'Obesity',
            'Parasites',
            'Scabies',
            'Salmonella',
            'Sexually Transmitted Diseases',
            'Stroke',
            'Traumatic Brain Injury',
            'TBI',
            'Trichomonas Infection',
            'Trichomoniasis',
            'Tuberculosis',
            'TB',
            'Zika Virus',
        ]

  #TODO: we need to get the title and pronoun list for a particular language in order to do better backtrans matching
  def __init__(self, target_lang='fr', ner_regexes=None, ontology=None, strip_chars=None, person_pronouns=None, other_pronouns=None, salt=""):
    super().__init__(ner_regexes=ner_regexes, ontology=ontology, strip_chars=strip_chars, person_pronouns=person_pronouns, other_pronouns=other_pronouns)
    for key, val in PIIProcessor.pii_ontology.items():
      self.ontology[key] = val
    self.backtrans = BackTranslate(target_lang)
    self.target_lang = target_lang
    try:
      self.stopwords_target_lang = {} if target_lang not in self.langs else set(stopwords.words(self.langs[target_lang].lower()))
    except:
      try:
        nltk.corpus.stopwords.words(self.langs[target_lang].lower())
        self.stopwords_target_lang = {} if target_lang not in self.langs else set(stopwords.words(self.langs[target_lang].lower()))
      except:
        self.stopwords_target_lang = {} 
    self.salt = salt
    if PIIProcessor.faker_en is None:
      PIIProcessor.faker_en = Faker(self.faker_map['en'])
      PIIProcessor.faker_en.add_provider(internet)
      PIIProcessor.faker_en.add_provider(geo)
      PIIProcessor.faker_en.add_provider(address)
      PIIProcessor.faker_en.add_provider(date_time)
      PIIProcessor.faker_en.add_provider(job)
      PIIProcessor.faker_en.add_provider(person)
      PIIProcessor.faker_en.add_provider(bank)
      PIIProcessor.faker_en.add_provider(credit_card)
      PIIProcessor.faker_en.add_provider(ssn)
    self.faker_target_lang = Faker(self.faker_map[target_lang])
    self.faker_target_lang.add_provider(internet)
    self.faker_target_lang.add_provider(geo)
    self.faker_target_lang.add_provider(address)
    self.faker_target_lang.add_provider(date_time)
    self.faker_target_lang.add_provider(job)
    self.faker_target_lang.add_provider(person)
    self.faker_target_lang.add_provider(bank)
    self.faker_target_lang.add_provider(credit_card)
    self.faker_target_lang.add_provider(ssn)
    if PIIProcessor.race_list == []:
      list2 = list(set(self.race_list0+person.Provider.language_names+list(self.backtrans.langs.values())))
      list2.remove('Interlingua')
      PIIProcessor.race_list = list(set(list2))
    #for now, we are not going to add 
    self.races_en = dict ([(word, "RACE") for word in self.race_list])
    self.titles_en = dict ([(word, "TITLE") for word in self.title2pronoun.keys()])
    self.pronoun_en = dict ([(word, "PRONOUN") for word in self.pronoun.keys()])
    self.gender_en = dict ([(word, "GENDER") for word in self.gender2pronoun.keys()])
    self.religion_en = dict ([(word, "RELIGION") for word in self.person2religion.keys()])
    self.disease_en = dict ([(word, "DISEASE") for word in self.disease_list])
    #jobs can be proxies for gender or race, so we may want to swap jobs. 
    #TODO: add in male_jobs and female_jobs
    self.job_en = dict ([(word.split(",")[0].strip() , "JOB") for word in job.Provider.jobs])
    
    #for religion, diease, and jobs, we may want to genericize instead of swap jobs because that may change the meaning of the sentence. 
    #e.g., the doctor did X. He xxxx => the health worker did X. She xxxx.

    #TODO: union membership
    #TODO: use gender_swap to swap genders.
    self.add_ontology(self.titles_en)
    self.add_ontology(self.pronoun_en)            
    self.add_ontology(self.gender_en)   
    self.add_ontology(self.religion_en)  
    self.add_ontology(self.disease_en)    
    self.add_ontology(self.job_en)    
    #self.add_ontology(self.nrp_recognizer) 
    # maps ner_label to function to generate fake value
    self.default_recognizer = dict([("PRONOUN", None), ("TITLE", None), ("GENDER", None), ("PERSON", None), \
                           ("CREDIT_CARD", None), ("CRYPTO", None), ("IP_ADDRESS", None), ("LOC", None), \
                           ("US_SSN", self.faker_en.ssn), ("TIME", None), ("GPE", self.faker_en.country), \
                           ("DATE", self.faker_en.date), ("STREET_ADDRESS", None), ("ORG", None), \
                           ("DOMAIN_NAME", None), ("JOB", self.faker_en.job), ("NORP", self.nrp), ("RACE", self.nrp), \
                           ("DISEASE", self.disease), ("RELIGION", self.religion), ("PHONE_NUMBER", None), ("EMAIL_ADDRESS", None)])
    
    if ner_regexes is None:
      ner_regexes = PIIProcessor.default_ner_regexes
    self.ner_regexes = ner_regexes

  def gender(self):
        return choice(list(self.gender2pronoun.keys()))

  def religion(self):
        return choice(list(self.per2onregligion.keys()))

  def disease(self):
        return choice(self.disease_list)

  def nrp(self):
        return choice(self.nrp_list2)

  @staticmethod
  def encrypt(s, salt=""):
    """ 
    we can add a salt to make it more secure
    what we mainly want to avoid is if there was a seurity incident and the dataset of names we might have gathered
    from a webcrawled or licensed dataset is exposed. 
    """
    return (base64.b64encode(hashlib.sha512((s.strip().lower()+salt).encode()).digest()))


  def add_PII_context_data_en(self, span, swap_fn, chunk2ner, chunk2ref, ref2chunk, PII_context, pronoun=None, args=None):
    """
    Modifies the PII_context by adding associations of old detected PII/NER labels to a mention/span. 
    :arg span: the old NER/II value in english that was detected.
    :arg label: the label detected
    :arg swap_fn: a function to apply to get a new value. we do the swapping in english and depend on the back-trans to get the word in target_lang.
    :arg PII_context:  maps (text_to_anonymize, label, target pronoun) = > val
      val is in the form of [id] or "[id] swap_val *" or "[id] origial_text *". 
      val is used as a replacement/anonymizing in english, to be translated to the target language. 
    :arg pronoun: the current target pronoun (he, she, they)

    Do gender swaping when required. otherwise, guess the pronoun from the pronoun, title, gender words or name. 
    If we can't guess the pronoun from the name, just use coref to get the the pronoun for the coref group, 
    and then spwap, pronoun, title, or gender word closest to the person's name.
    
    """
    do_gender_swap=True
    do_religion_swap=True
    do_job_swap=True
    do_nrp_swap=True
    if args is not None:
      do_gender_swap=ars.do_gender_swap
      do_religion_swap=args.do_religion_swap
      do_job_swap=args.do_job_swap
      do_nrp_swap=ars.do_nrp_swap
    if pronoun is None:
      pronoun = {}
    word = span[0]
    label, coref = chunk2ner.get(span), chunk2ref.get(span)
    id = len(PII_context)+1
    val = f"[{id}]"
    if label == "PRONOUN":
      if do_gender_swap:
        if pronoun.get(coref):
          target_pronoun = pronoun[coref] # this might be wrong. we need to map "she" back to "her"
        else:
          target_pronoun = self.pronoun_swap.get(word, "she")
          pronoun[coref] = self.pronoun.get(target_pronoun, "she").lower()
        return target_pronoun
      else:
        return word
    elif label == "TITLE":
      if do_gender_swap:
        if pronoun.get(coref):
          target_title = self.pronoun2itle[pronoun[coref]]
        else:
          target_title = self.title_swap.get(word.lower(), "Mrs.")
          pronoun[coref] = self.title2pronoun.get(target_title, "she")
        return target_title
      else:
        return word
    elif label == "GENDER":
      if do_gender_swap:
        if pronoun.get(coref):
          gender = self.pronoun2gender[pronoun[coref]]
        else:
          gender = self.gender()
        pronoun[coref] = self.gender2pronoun.get(gender, "she")
      else:
        gender = word
      val = f"[{id}] {gender} * "  
    elif label == "PERSON":
      if pronoun:
        pr = pronoun[coref]
      else:
        pr = "she"
      pronoun[coref] = pr
      val =  f"[{id}]"

    target_pronoun = None
    if label in ("PERSON", "GENDER",):
      target_pronoun = None if not pronoun else pronoun.get(coref, "she")
    if swap_fn is not None:
      swap_val = swap_fn()
      val = f"[{id}] {swap_val} * "
    key = (word, label, target_pronoun)
    if key in PII_context:
      return PII_context[key] 
    key = (word, label, None)
    if key in PII_context:
      if target_pronoun is not None:
        val = PII_context[key]
        del PII_context[key]
        key = (word, label, target_pronoun)
        PII_context[key] = val
      return PII_context[key] 
    if " " in word:
      xArr = word.split()
      key = (xArr[0], label)
      if key in PII_context:
        return PII_context[key] 
      key = (xArr[-1], label)
      if key in PII_context:
        return PII_context[key]
      key = (word, label, target_pronoun)
      PII_context[key] = val
      xArr = word.split()
      for x in xArr:
        key = (x, label, target_pronoun)
        PII_context[key] = val
    else:
      key = (word, label, target_pronoun)
      PII_context[key] = val
    return PII_context[key]

  #check if faker.name() does proportionate male/female or just picks at random in the firstname, lastname list.
  def apply_PII_en(self, analyzer_results, PII_context={}, pronoun=None, recognizer=None, sep=" "):  
    """
    Apply the recognizer to the NER labeled span to process the span appropriately. 
    """
    text = []
    if pronoun is None:
      pronoun = {}
    if recognizer is None:
      recognizer = self.default_recognizer
    chunk2ner = analyzer_results['chunk2ner']
    for span in  analyzer_results['chunks']:
      label = chunk2ner.get(span, '')
      if label in ('', 'NOUN', None):
        text.append(span[0])
      else:
        if label not in recognizer:
          text.append(span[0])
        else:
          text.append(self.add_PII_context_data_en(span, recognizer[label], analyzer_results['chunk2ner'], analyzer_results['chunk2ref'], analyzer_results['ref2chunk'], PII_context, pronoun))
      return sep.join(text)

  def apply_PII_target_lang(self, text_to_anonymize, PII_context):
      """
      Apply PII replacement in the target language. Some PII can't easily be translated back, or don't need to be translated back,
      so we apply PII swapping in the target language. 

      Uses target_pronoun to determine the generated name of the person.
      For person, replace the first occurance with the full name, and subsequent occurance with a short version of the name.
      """
      target_lang = self.target_lang
      for key, target in PII_context.items():
        orig_val, label, target_pronoun = key
        new_val = target
        complex_target = False
        if "*" in target:
          _, new_val = target.split("]")
          new_val = new_val.strip(" *")
          complex_target = True
        if label == "PERSON":
          if target_pronoun is not None:
            if target_pronoun == "she":
              new_val = self.faker_target_lang.name_female() 
            if target_pronoun == "he":
              new_val = self.faker_target_lang.name_male() 
            else:
              new_val = self.faker_target_lang.name() 
          else:
            new_val = self.faker_target_lang.name() 
        elif label == "PHONE_NUMBER":
          new_val = self.faker_target_lang.phone_number()
        elif label == "EMAIL_ADDRESS":
          new_val = self.faker_target_lang.safe_email()
        elif label == "DOMAIN_NAME":
          new_val = self.faker_target_lang.hostname()
        elif label == "STREET_ADDRESS": # TODO, make the location close to orig_val
          if random.randint(0,3)==0:
            new_val = self.faker_target_lang.address()       
          elif random.randint(0,3)==0:
            new_val = self.faker_target_lang.country()  
          else:
            new_val = self.faker_target_lang.city() 
        if label == "PERSON":
          if target_lang in ("zh", "ja", "ko"):
            text_to_anonymize = text_to_anonymize.replace(target, new_val, 1)
            new_val = new_val[0]
          elif " " in new_val:
            text_to_anonymize = text_to_anonymize.replace(target, new_val, 1)
            new_val = new_val.split()[-1]         
        text_to_anonymize = text_to_anonymize.replace(target, new_val)
      return text_to_anonymize

  @staticmethod
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

  def process(self, text="", batch=None, row_id=0, doc_id=0, lang=None, encrypt_value=False, return_original_text=False):
    """
    The main PII processor.
    """
    if batch is None:
      batch = [text]
    if encrypt_value and return_original_text:
      print ("warning: returning original text, while encrypting a person's info will still expose the person's info.")
    if lang is None:
      lang = self.target_lang
    if self.target_lang in ("ja", "ko", "zh"):
      sep = ""
    else:
      sep = " "
    if lang not in (self.target_lang, 'en'):
      raise RuntimeError(f"can only process text in the target language {self.target_lang} or convert from English to the target language.")
    # we do translation to english because the tools we use work in english mostly. we translate back to target language at the end.  
    ret = []
    for text in batch:
      if lang == 'en':
        text_to_anonymize_en = text
      else:
        text_to_anonymize_en = self.backtrans.translate(text, fr=lang, to='en')
      analyzer_results = self.analyze_with_ner_coref_en(text=text_to_anonymize_en, row_id=row_id, doc_id=doc_id)
      PII_context = {} 
      pronoun = {}
      print (analyzer_results)
      pii_text_analyzed_en = self.apply_PII_en(analyzer_results,  PII_context=PII_context, pronoun=pronoun) 
      print (pii_text_analyzed_en)
      if self.target_lang=='en':
        templated_text = pii_text_analyzed_en
      else:
        templated_text =  self.backtrans.translate(pii_text_analyzed_en, fr='en', to=self.target_lang)
      if lang == 'en':
        anonymized_text = self.apply_PII_target_lang(templated_text, PII_context)
        target_to_label_ret = {}
        for key, target in PII_context.items():
          if encrypt_value:
            target_to_label_ret[target]  = {'ecnrypted_text_en': self.encrypt(key[0], self.salt), 'label': key[1]}
          else:
            target_to_label_ret[target]  = {'text_en': key[0],  'label': key[1]}
        if return_original_text:
          ret.append({"text": text, f"anonymized_text_{self.target_lang}": anonymized_text, f"template_{self.target_lang}": templated_text,  "pii": target_to_label_ret})
        else:
          ret.append({f"anonymized_text_{self.target_lang}": anonymized_text, f"template_{self.target_lang}": templated_text,  "pii": target_to_label_ret})
        row_id += 1
        continue
      else:
        target_to_label = dict([(b, a) for a, b in PII_context.items()])
        target_to_label_ret={}
        (blocks2, score) =  self.get_aligned_text(text, anonymized_text0, self.target_lang)
        #since we know what slots we are replacing, 
        #we can align with the original sentence in the original language, 
        #and then extract the original NER value, or a translated version of the new value. 
        for (s1, s2, matched) in blocks2:    
            if "[" in s2 and "]" in s2:
              _id, orig_value = s2.split("[")[1].split("]")
              _id = _id.strip()
              orig_value = orig_value.strip(" *")
              key = f"[{_id}] {orig_value} *"
              ner_label = target_to_label.get(key, [None, orig_value])[1]
              if not _id or _id[0] not in "0123456789": continue
              if key not in target_to_label_ret:
                # we might be able to also capture the english translated value.
                if encrypt_value:
                  target_to_label_ret[key]  = {f'encrypted_text_{self.target_lang}': self.encrypt(s1, self.salt), 'ner_label': ner_label}
                else:
                  target_to_label_ret[key]  = {f'text_{self.target_lang}': s1,  'ner_label': ner_label}
            #elif len(s2) <= 3: # this should actualy be a check if s2 is a pronoun or title that has been swapped?
            #  print ('swapping short text', s1, s2)

        templated_text = templated_text.strip()
        anonymized_text = self.apply_PII_target_lang(templated_text, PII_context)
        if return_original_text:
          ret.append({"text": text, f"anonymized_text_{self.target_lang}": anonymized_text, f"template_{self.target_lang}": templated_text,  "pii": target_to_label_ret})
        else:
          ret.append({f"anonymized_text_{self.target_lang}": anonymized_text, f"template_{self.target_lang}": templated_text, "pii": target_to_label_ret})
        row_id += 1
        continue
    return ret
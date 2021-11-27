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
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, MarianMTModel, AutoTokenizer, pipeline
import torch
import sys
from tqdm import tqdm

stopwords_en = set(stopwords.words('english'))

junk_dict = dict([(a, 1) for a in "' 0123456789¯_§½¼¾×|†—~\"—±′–'°−{}[]·-\'?,./<>!@#^&*()+-‑=:;`→¶'"])

# from https://github.com/madisonmay/CommonRegex/blob/master/commonregex.py which is under the MIT License
# Phone is not working. email is not working?
basic_regex = [
    ("EMAIL_ADDRESS", re.compile(
        "([a-z0-9!#$%&'*+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)")),
    ("AGE", re.compile("\S+ years old|\S+\-years\-old|\S+ year old|\S+\-year\-old")),
    # ("PHONE_NUMBER"    , re.compile('((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))')),
    # ("PHONE_NUMBER"    , re.compile('((?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|(?:[2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?(?:[2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?(?:[0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(?:\d+)?))')),
    ("STREET_ADDRESS", re.compile(
        '\d{1,4} [\w\s]{1,20} (?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\W?(?=\s|$)')),
    ("STREET_ADDRESS", re.compile('\b\d{5}(?:[-\s]\d{4})?\b')),
    ("STREET_ADDRESS", re.compile('P\.? ?O\.? Box \d+')),
    ("GOVT_ID", re.compile(
        '(?!000|666|333)0*(?:[0-6][0-9][0-9]|[0-7][0-6][0-9]|[0-7][0-7][0-2])[- ](?!00)[0-9]{2}[- ](?!0000)[0-9]{4}')),
    ("DISEASE", re.compile("diabetes|cancer|HIV|AIDS|Alzheimer's|Alzheimer|heart disease")),
    ("NORP", re.compile("upper class|middle class|working class|lower class")),
]

# quick and dirty lexicon management. we could use wikipedia to find some of these words, but I don't have time to do this. 
# A more general version of the code using wikipedia, ontology, etc. is in the pii_processing/pii and onology folder.

country = {'Zimbabwe', 'Panama', 'the Soviet Union', 'Bahamas''Colombia', 'AMERICA', 'Norway',
           'The United States of America', 'Indonesia', 'Switzerland', 'U.S.A.', 'Chile', 'U.K.', "North Korea's",
           'USSR', 'Greece', 'N. Korea', 'Viet Nam', 'america', 'U.S', 'South Korea', 'Sweden', 'New Zealand',
           'The United States', 'Pakistan', 'United States', 'Holland', 'Taiwan', 'Ukraine', 'Italy', 'South Africa',
           'Turkey', 'Spain', 'the United States of America', 'Britain', 'Saudi Arabia', 'Argentina', 'Vietnam', 'Cuba',
           'Syria', 'UK', 'Belgium', 'Iran', 'Latvia', 'India', 'France', 'Great Britain', 'El Salvador', 'Australia',
           'Brazil', 'Venezuela', 'England', 'the Roman Empire', 'russia', 'Gaza', 'Costa Rica', 'Yugoslavia',
           'the United Kingdom', 'the United States', 'USA', 'North Korea', 'U.S.', 'Canada', 'US', 'America', 'China',
           'U.S.', 'Russia', 'Israel', 'Afghanistan', 'PRC', 'Mexico', 'Iraq', 'Japan', 'Germany', }
public_figures = {'Alec Baldwin', 'Stephen Harper', 'Al Gore', 'tRump', 'Trump', 'Harper', 'Justin', 'Comey', 'Obama',
                  'Gary Johnson', 'Lindsey Graham', 'Michelle Obama', 'Mark Twain', 'Jared Kushner', 'Vladimir Putin',
                  'Putin', 'Osama bin Laden', 'Fidel Castro', 'Mitch McConnell', 'Elizabeth Warren', 'Hilary Clinton',
                  'George Bush', 'John Paul II', 'Jimmy Carter', "Hillary Clinton's", 'Ann Coulter', 'Rush Limbaugh',
                  'JFK', 'Richard Nixon', 'Cameron Sellers', 'Ted Cruz', 'Richard Nixon', 'Jill Stein', 'Brad Wall',
                  'James Comey', 'Jeff Sessions', 'Loretta Lynch', 'Sean Spicer', 'Nancy Pelosi', 'Ronald Reagan',
                  'Gray Davis', 'Donald J. Trump', 'Dan Sullivan', 'George W. Bush', 'George Soros', 'Lisa Murkowski',
                  'Gore', 'Kim Jong Un', "Donald Trump's", 'Steve Bannon', 'Alex Jones', 'Paul Ryan', 'Trumps',
                  'Clintons', 'John McCain', 'Hillary', 'Obama', 'Trump', 'Clinton', 'Trudeau', 'Donald Trump', 'Bush',
                  'Jesus', 'Hillary Clinton', 'Justin Trudeau', 'Bernie Sanders', 'Reagan', 'Bill Clinton',
                  'Pope Francis', 'Hitler', 'Nixon', 'Sarah Palin', 'Barack Obama', }
NORP = {'Islam', 'Conservative', 'the Conservative Party', 'GOP', 'Christianity', 'Church', 'the Republican Party',
        'the Democratic Party', 'the Catholic Church', 'the Liberal Party', 'the Tea Party', 'Republican Party',
        'the Roman Catholic Church', 'UNION', 'MAGA', 'the Democrat Party', }
swap_org = ['Ford', 'Facebook', 'CNN', 'ABC', 'NBC', 'CBS', 'Google', 'Amazon', 'AOL', 'NCR', 'GE', 'Enron']
bantu_surnames = ["Dlamini", "Gumede", "Hadebe", "Ilunga", "Kamau", "Khoza", "Lubega", "M'Bala", "Mabaso", "Mabika",
                  "Mabizela", "Mabunda", "Mabuza", "Macharia", "Madima", "Madondo", "Mahlangu", "Maidza", "Makhanya",
                  "Malewezi", "Mamba", "Mandanda", "Mandlate", "Mangwana", "Manjate", "Maponyane", "Mapunda", "Maraire",
                  "Masango", "Maseko", "Masemola", "Masengo", "Mashabane", "Masire", "Masondo", "Masuku", "Mataka",
                  "Matovu", "Mbala", "Mbatha", "Mbugua", "Mchunu", "Mkhize", "Mofokeng", "Mokonyane", "Mutombo",
                  "Ncube", "Ndagire", "Ndhlovu", "Ndikumana", "Ndiritu", "Ndlovu", "Ndzinisa", "Ngcobo", "Nkomo",
                  "Nkosi", "Nkurunziza", "Radebe", "Tshabalala", "Tshivhumbe", "Vila"]

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

mariam_mt_mapping = {('aav', 'en'): 'Helsinki-NLP/opus-mt-aav-en', ('aed', 'es'): 'Helsinki-NLP/opus-mt-aed-es',
                     ('af', 'de'): 'Helsinki-NLP/opus-mt-af-de', ('af', 'en'): 'Helsinki-NLP/opus-mt-af-en',
                     ('af', 'eo'): 'Helsinki-NLP/opus-mt-af-eo', ('af', 'es'): 'Helsinki-NLP/opus-mt-af-es',
                     ('af', 'fi'): 'Helsinki-NLP/opus-mt-af-fi', ('af', 'fr'): 'Helsinki-NLP/opus-mt-af-fr',
                     ('af', 'nl'): 'Helsinki-NLP/opus-mt-af-nl', ('af', 'ru'): 'Helsinki-NLP/opus-mt-af-ru',
                     ('af', 'sv'): 'Helsinki-NLP/opus-mt-af-sv', ('afa', 'afa'): 'Helsinki-NLP/opus-mt-afa-afa',
                     ('afa', 'en'): 'Helsinki-NLP/opus-mt-afa-en', ('alv', 'en'): 'Helsinki-NLP/opus-mt-alv-en',
                     ('am', 'sv'): 'Helsinki-NLP/opus-mt-am-sv', ('ar', 'de'): 'Helsinki-NLP/opus-mt-ar-de',
                     ('ar', 'el'): 'Helsinki-NLP/opus-mt-ar-el', ('ar', 'en'): 'Helsinki-NLP/opus-mt-ar-en',
                     ('ar', 'eo'): 'Helsinki-NLP/opus-mt-ar-eo', ('ar', 'es'): 'Helsinki-NLP/opus-mt-ar-es',
                     ('ar', 'fr'): 'Helsinki-NLP/opus-mt-ar-fr', ('ar', 'he'): 'Helsinki-NLP/opus-mt-ar-he',
                     ('ar', 'it'): 'Helsinki-NLP/opus-mt-ar-it', ('ar', 'pl'): 'Helsinki-NLP/opus-mt-ar-pl',
                     ('ar', 'ru'): 'Helsinki-NLP/opus-mt-ar-ru', ('ar', 'tr'): 'Helsinki-NLP/opus-mt-ar-tr',
                     ('art', 'en'): 'Helsinki-NLP/opus-mt-art-en', ('ase', 'de'): 'Helsinki-NLP/opus-mt-ase-de',
                     ('ase', 'en'): 'Helsinki-NLP/opus-mt-ase-en', ('ase', 'es'): 'Helsinki-NLP/opus-mt-ase-es',
                     ('ase', 'fr'): 'Helsinki-NLP/opus-mt-ase-fr', ('ase', 'sv'): 'Helsinki-NLP/opus-mt-ase-sv',
                     ('az', 'en'): 'Helsinki-NLP/opus-mt-az-en', ('az', 'es'): 'Helsinki-NLP/opus-mt-az-es',
                     ('az', 'tr'): 'Helsinki-NLP/opus-mt-az-tr', ('bat', 'en'): 'Helsinki-NLP/opus-mt-bat-en',
                     ('bcl', 'de'): 'Helsinki-NLP/opus-mt-bcl-de', ('bcl', 'en'): 'Helsinki-NLP/opus-mt-bcl-en',
                     ('bcl', 'es'): 'Helsinki-NLP/opus-mt-bcl-es', ('bcl', 'fi'): 'Helsinki-NLP/opus-mt-bcl-fi',
                     ('bcl', 'fr'): 'Helsinki-NLP/opus-mt-bcl-fr', ('bcl', 'sv'): 'Helsinki-NLP/opus-mt-bcl-sv',
                     ('be', 'es'): 'Helsinki-NLP/opus-mt-be-es', ('bem', 'en'): 'Helsinki-NLP/opus-mt-bem-en',
                     ('bem', 'es'): 'Helsinki-NLP/opus-mt-bem-es', ('bem', 'fi'): 'Helsinki-NLP/opus-mt-bem-fi',
                     ('bem', 'fr'): 'Helsinki-NLP/opus-mt-bem-fr', ('bem', 'sv'): 'Helsinki-NLP/opus-mt-bem-sv',
                     ('ber', 'en'): 'Helsinki-NLP/opus-mt-ber-en', ('ber', 'es'): 'Helsinki-NLP/opus-mt-ber-es',
                     ('ber', 'fr'): 'Helsinki-NLP/opus-mt-ber-fr', ('bg', 'de'): 'Helsinki-NLP/opus-mt-bg-de',
                     ('bg', 'en'): 'Helsinki-NLP/opus-mt-bg-en', ('bg', 'eo'): 'Helsinki-NLP/opus-mt-bg-eo',
                     ('bg', 'es'): 'Helsinki-NLP/opus-mt-bg-es', ('bg', 'fi'): 'Helsinki-NLP/opus-mt-bg-fi',
                     ('bg', 'fr'): 'Helsinki-NLP/opus-mt-bg-fr', ('bg', 'it'): 'Helsinki-NLP/opus-mt-bg-it',
                     ('bg', 'ru'): 'Helsinki-NLP/opus-mt-bg-ru', ('bg', 'sv'): 'Helsinki-NLP/opus-mt-bg-sv',
                     ('bg', 'tr'): 'Helsinki-NLP/opus-mt-bg-tr', ('bg', 'uk'): 'Helsinki-NLP/opus-mt-bg-uk',
                     ('bi', 'en'): 'Helsinki-NLP/opus-mt-bi-en', ('bi', 'es'): 'Helsinki-NLP/opus-mt-bi-es',
                     ('bi', 'fr'): 'Helsinki-NLP/opus-mt-bi-fr', ('bi', 'sv'): 'Helsinki-NLP/opus-mt-bi-sv',
                     ('bn', 'en'): 'Helsinki-NLP/opus-mt-bn-en', ('bnt', 'en'): 'Helsinki-NLP/opus-mt-bnt-en',
                     ('bzs', 'en'): 'Helsinki-NLP/opus-mt-bzs-en', ('bzs', 'es'): 'Helsinki-NLP/opus-mt-bzs-es',
                     ('bzs', 'fi'): 'Helsinki-NLP/opus-mt-bzs-fi', ('bzs', 'fr'): 'Helsinki-NLP/opus-mt-bzs-fr',
                     ('bzs', 'sv'): 'Helsinki-NLP/opus-mt-bzs-sv', ('ca', 'de'): 'Helsinki-NLP/opus-mt-ca-de',
                     ('ca', 'en'): 'Helsinki-NLP/opus-mt-ca-en', ('ca', 'es'): 'Helsinki-NLP/opus-mt-ca-es',
                     ('ca', 'fr'): 'Helsinki-NLP/opus-mt-ca-fr', ('ca', 'it'): 'Helsinki-NLP/opus-mt-ca-it',
                     ('ca', 'nl'): 'Helsinki-NLP/opus-mt-ca-nl', ('ca', 'pt'): 'Helsinki-NLP/opus-mt-ca-pt',
                     ('ca', 'uk'): 'Helsinki-NLP/opus-mt-ca-uk', ('cau', 'en'): 'Helsinki-NLP/opus-mt-cau-en',
                     ('ccs', 'en'): 'Helsinki-NLP/opus-mt-ccs-en', ('ceb', 'en'): 'Helsinki-NLP/opus-mt-ceb-en',
                     ('ceb', 'es'): 'Helsinki-NLP/opus-mt-ceb-es', ('ceb', 'fi'): 'Helsinki-NLP/opus-mt-ceb-fi',
                     ('ceb', 'fr'): 'Helsinki-NLP/opus-mt-ceb-fr', ('ceb', 'sv'): 'Helsinki-NLP/opus-mt-ceb-sv',
                     ('cel', 'en'): 'Helsinki-NLP/opus-mt-cel-en', ('chk', 'en'): 'Helsinki-NLP/opus-mt-chk-en',
                     ('chk', 'es'): 'Helsinki-NLP/opus-mt-chk-es', ('chk', 'fr'): 'Helsinki-NLP/opus-mt-chk-fr',
                     ('chk', 'sv'): 'Helsinki-NLP/opus-mt-chk-sv', ('cpf', 'en'): 'Helsinki-NLP/opus-mt-cpf-en',
                     ('cpp', 'cpp'): 'Helsinki-NLP/opus-mt-cpp-cpp', ('cpp', 'en'): 'Helsinki-NLP/opus-mt-cpp-en',
                     ('crs', 'de'): 'Helsinki-NLP/opus-mt-crs-de', ('crs', 'en'): 'Helsinki-NLP/opus-mt-crs-en',
                     ('crs', 'es'): 'Helsinki-NLP/opus-mt-crs-es', ('crs', 'fi'): 'Helsinki-NLP/opus-mt-crs-fi',
                     ('crs', 'fr'): 'Helsinki-NLP/opus-mt-crs-fr', ('crs', 'sv'): 'Helsinki-NLP/opus-mt-crs-sv',
                     ('cs', 'de'): 'Helsinki-NLP/opus-mt-cs-de', ('cs', 'en'): 'Helsinki-NLP/opus-mt-cs-en',
                     ('cs', 'eo'): 'Helsinki-NLP/opus-mt-cs-eo', ('cs', 'fi'): 'Helsinki-NLP/opus-mt-cs-fi',
                     ('cs', 'fr'): 'Helsinki-NLP/opus-mt-cs-fr', ('cs', 'sv'): 'Helsinki-NLP/opus-mt-cs-sv',
                     ('cs', 'uk'): 'Helsinki-NLP/opus-mt-cs-uk', ('csg', 'es'): 'Helsinki-NLP/opus-mt-csg-es',
                     ('csn', 'es'): 'Helsinki-NLP/opus-mt-csn-es', ('cus', 'en'): 'Helsinki-NLP/opus-mt-cus-en',
                     ('cy', 'en'): 'Helsinki-NLP/opus-mt-cy-en', ('da', 'de'): 'Helsinki-NLP/opus-mt-da-de',
                     ('da', 'en'): 'Helsinki-NLP/opus-mt-da-en', ('da', 'eo'): 'Helsinki-NLP/opus-mt-da-eo',
                     ('da', 'es'): 'Helsinki-NLP/opus-mt-da-es', ('da', 'fi'): 'Helsinki-NLP/opus-mt-da-fi',
                     ('da', 'fr'): 'Helsinki-NLP/opus-mt-da-fr', ('da', 'no'): 'Helsinki-NLP/opus-mt-da-no',
                     ('da', 'ru'): 'Helsinki-NLP/opus-mt-da-ru', ('de', 'ZH'): 'Helsinki-NLP/opus-mt-de-ZH',
                     ('de', 'af'): 'Helsinki-NLP/opus-mt-de-af', ('de', 'ar'): 'Helsinki-NLP/opus-mt-de-ar',
                     ('de', 'ase'): 'Helsinki-NLP/opus-mt-de-ase', ('de', 'bcl'): 'Helsinki-NLP/opus-mt-de-bcl',
                     ('de', 'bg'): 'Helsinki-NLP/opus-mt-de-bg', ('de', 'bi'): 'Helsinki-NLP/opus-mt-de-bi',
                     ('de', 'bzs'): 'Helsinki-NLP/opus-mt-de-bzs', ('de', 'ca'): 'Helsinki-NLP/opus-mt-de-ca',
                     ('de', 'crs'): 'Helsinki-NLP/opus-mt-de-crs', ('de', 'cs'): 'Helsinki-NLP/opus-mt-de-cs',
                     ('de', 'da'): 'Helsinki-NLP/opus-mt-de-da', ('de', 'de'): 'Helsinki-NLP/opus-mt-de-de',
                     ('de', 'ee'): 'Helsinki-NLP/opus-mt-de-ee', ('de', 'efi'): 'Helsinki-NLP/opus-mt-de-efi',
                     ('de', 'el'): 'Helsinki-NLP/opus-mt-de-el', ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en',
                     ('de', 'eo'): 'Helsinki-NLP/opus-mt-de-eo', ('de', 'es'): 'Helsinki-NLP/opus-mt-de-es',
                     ('de', 'et'): 'Helsinki-NLP/opus-mt-de-et', ('de', 'eu'): 'Helsinki-NLP/opus-mt-de-eu',
                     ('de', 'fi'): 'Helsinki-NLP/opus-mt-de-fi', ('de', 'fj'): 'Helsinki-NLP/opus-mt-de-fj',
                     ('de', 'fr'): 'Helsinki-NLP/opus-mt-de-fr', ('de', 'gaa'): 'Helsinki-NLP/opus-mt-de-gaa',
                     ('de', 'gil'): 'Helsinki-NLP/opus-mt-de-gil', ('de', 'guw'): 'Helsinki-NLP/opus-mt-de-guw',
                     ('de', 'ha'): 'Helsinki-NLP/opus-mt-de-ha', ('de', 'he'): 'Helsinki-NLP/opus-mt-de-he',
                     ('de', 'hil'): 'Helsinki-NLP/opus-mt-de-hil', ('de', 'ho'): 'Helsinki-NLP/opus-mt-de-ho',
                     ('de', 'hr'): 'Helsinki-NLP/opus-mt-de-hr', ('de', 'ht'): 'Helsinki-NLP/opus-mt-de-ht',
                     ('de', 'hu'): 'Helsinki-NLP/opus-mt-de-hu', ('de', 'ig'): 'Helsinki-NLP/opus-mt-de-ig',
                     ('de', 'ilo'): 'Helsinki-NLP/opus-mt-de-ilo', ('de', 'is'): 'Helsinki-NLP/opus-mt-de-is',
                     ('de', 'iso'): 'Helsinki-NLP/opus-mt-de-iso', ('de', 'it'): 'Helsinki-NLP/opus-mt-de-it',
                     ('de', 'kg'): 'Helsinki-NLP/opus-mt-de-kg', ('de', 'ln'): 'Helsinki-NLP/opus-mt-de-ln',
                     ('de', 'loz'): 'Helsinki-NLP/opus-mt-de-loz', ('de', 'lt'): 'Helsinki-NLP/opus-mt-de-lt',
                     ('de', 'lua'): 'Helsinki-NLP/opus-mt-de-lua', ('de', 'ms'): 'Helsinki-NLP/opus-mt-de-ms',
                     ('de', 'mt'): 'Helsinki-NLP/opus-mt-de-mt', ('de', 'niu'): 'Helsinki-NLP/opus-mt-de-niu',
                     ('de', 'nl'): 'Helsinki-NLP/opus-mt-de-nl', ('de', 'no'): 'Helsinki-NLP/opus-mt-de-no',
                     ('de', 'nso'): 'Helsinki-NLP/opus-mt-de-nso', ('de', 'ny'): 'Helsinki-NLP/opus-mt-de-ny',
                     ('de', 'pag'): 'Helsinki-NLP/opus-mt-de-pag', ('de', 'pap'): 'Helsinki-NLP/opus-mt-de-pap',
                     ('de', 'pis'): 'Helsinki-NLP/opus-mt-de-pis', ('de', 'pl'): 'Helsinki-NLP/opus-mt-de-pl',
                     ('de', 'pon'): 'Helsinki-NLP/opus-mt-de-pon', ('de', 'tl'): 'Helsinki-NLP/opus-mt-de-tl',
                     ('de', 'uk'): 'Helsinki-NLP/opus-mt-de-uk', ('de', 'vi'): 'Helsinki-NLP/opus-mt-de-vi',
                     ('dra', 'en'): 'Helsinki-NLP/opus-mt-dra-en', ('ee', 'de'): 'Helsinki-NLP/opus-mt-ee-de',
                     ('ee', 'en'): 'Helsinki-NLP/opus-mt-ee-en', ('ee', 'es'): 'Helsinki-NLP/opus-mt-ee-es',
                     ('ee', 'fi'): 'Helsinki-NLP/opus-mt-ee-fi', ('ee', 'fr'): 'Helsinki-NLP/opus-mt-ee-fr',
                     ('ee', 'sv'): 'Helsinki-NLP/opus-mt-ee-sv', ('efi', 'de'): 'Helsinki-NLP/opus-mt-efi-de',
                     ('efi', 'en'): 'Helsinki-NLP/opus-mt-efi-en', ('efi', 'fi'): 'Helsinki-NLP/opus-mt-efi-fi',
                     ('efi', 'fr'): 'Helsinki-NLP/opus-mt-efi-fr', ('efi', 'sv'): 'Helsinki-NLP/opus-mt-efi-sv',
                     ('el', 'ar'): 'Helsinki-NLP/opus-mt-el-ar', ('el', 'eo'): 'Helsinki-NLP/opus-mt-el-eo',
                     ('el', 'fi'): 'Helsinki-NLP/opus-mt-el-fi', ('el', 'fr'): 'Helsinki-NLP/opus-mt-el-fr',
                     ('el', 'sv'): 'Helsinki-NLP/opus-mt-el-sv', ('en', 'aav'): 'Helsinki-NLP/opus-mt-en-aav',
                     ('en', 'af'): 'Helsinki-NLP/opus-mt-en-af', ('en', 'afa'): 'Helsinki-NLP/opus-mt-en-afa',
                     ('en', 'alv'): 'Helsinki-NLP/opus-mt-en-alv', ('en', 'ar'): 'Helsinki-NLP/opus-mt-en-ar',
                     ('en', 'az'): 'Helsinki-NLP/opus-mt-en-az', ('en', 'bat'): 'Helsinki-NLP/opus-mt-en-bat',
                     ('en', 'bcl'): 'Helsinki-NLP/opus-mt-en-bcl', ('en', 'bem'): 'Helsinki-NLP/opus-mt-en-bem',
                     ('en', 'ber'): 'Helsinki-NLP/opus-mt-en-ber', ('en', 'bg'): 'Helsinki-NLP/opus-mt-en-bg',
                     ('en', 'bi'): 'Helsinki-NLP/opus-mt-en-bi', ('en', 'bnt'): 'Helsinki-NLP/opus-mt-en-bnt',
                     ('en', 'bzs'): 'Helsinki-NLP/opus-mt-en-bzs', ('en', 'ca'): 'Helsinki-NLP/opus-mt-en-ca',
                     ('en', 'ceb'): 'Helsinki-NLP/opus-mt-en-ceb', ('en', 'cel'): 'Helsinki-NLP/opus-mt-en-cel',
                     ('en', 'chk'): 'Helsinki-NLP/opus-mt-en-chk', ('en', 'cpf'): 'Helsinki-NLP/opus-mt-en-cpf',
                     ('en', 'cpp'): 'Helsinki-NLP/opus-mt-en-cpp', ('en', 'crs'): 'Helsinki-NLP/opus-mt-en-crs',
                     ('en', 'cs'): 'Helsinki-NLP/opus-mt-en-cs', ('en', 'cus'): 'Helsinki-NLP/opus-mt-en-cus',
                     ('en', 'cy'): 'Helsinki-NLP/opus-mt-en-cy', ('en', 'da'): 'Helsinki-NLP/opus-mt-en-da',
                     ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de', ('en', 'dra'): 'Helsinki-NLP/opus-mt-en-dra',
                     ('en', 'ee'): 'Helsinki-NLP/opus-mt-en-ee', ('en', 'efi'): 'Helsinki-NLP/opus-mt-en-efi',
                     ('en', 'el'): 'Helsinki-NLP/opus-mt-en-el', ('en', 'eo'): 'Helsinki-NLP/opus-mt-en-eo',
                     ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es', ('en', 'et'): 'Helsinki-NLP/opus-mt-en-et',
                     ('en', 'eu'): 'Helsinki-NLP/opus-mt-en-eu', ('en', 'euq'): 'Helsinki-NLP/opus-mt-en-euq',
                     ('en', 'fi'): 'Helsinki-NLP/opus-mt-en-fi', ('en', 'fiu'): 'Helsinki-NLP/opus-mt-en-fiu',
                     ('en', 'fj'): 'Helsinki-NLP/opus-mt-en-fj', ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
                     ('en', 'ga'): 'Helsinki-NLP/opus-mt-en-ga', ('en', 'gaa'): 'Helsinki-NLP/opus-mt-en-gaa',
                     ('en', 'gem'): 'Helsinki-NLP/opus-mt-en-gem', ('en', 'gil'): 'Helsinki-NLP/opus-mt-en-gil',
                     ('en', 'gl'): 'Helsinki-NLP/opus-mt-en-gl', ('en', 'gmq'): 'Helsinki-NLP/opus-mt-en-gmq',
                     ('en', 'gmw'): 'Helsinki-NLP/opus-mt-en-gmw', ('en', 'grk'): 'Helsinki-NLP/opus-mt-en-grk',
                     ('en', 'guw'): 'Helsinki-NLP/opus-mt-en-guw', ('en', 'gv'): 'Helsinki-NLP/opus-mt-en-gv',
                     ('en', 'ha'): 'Helsinki-NLP/opus-mt-en-ha', ('en', 'he'): 'Helsinki-NLP/opus-mt-en-he',
                     ('en', 'hi'): 'Helsinki-NLP/opus-mt-en-hi', ('en', 'hil'): 'Helsinki-NLP/opus-mt-en-hil',
                     ('en', 'ho'): 'Helsinki-NLP/opus-mt-en-ho', ('en', 'ht'): 'Helsinki-NLP/opus-mt-en-ht',
                     ('en', 'hu'): 'Helsinki-NLP/opus-mt-en-hu', ('en', 'hy'): 'Helsinki-NLP/opus-mt-en-hy',
                     ('en', 'id'): 'Helsinki-NLP/opus-mt-en-id', ('en', 'ig'): 'Helsinki-NLP/opus-mt-en-ig',
                     ('en', 'iir'): 'Helsinki-NLP/opus-mt-en-iir', ('en', 'ilo'): 'Helsinki-NLP/opus-mt-en-ilo',
                     ('en', 'inc'): 'Helsinki-NLP/opus-mt-en-inc', ('en', 'ine'): 'Helsinki-NLP/opus-mt-en-ine',
                     ('en', 'is'): 'Helsinki-NLP/opus-mt-en-is', ('en', 'iso'): 'Helsinki-NLP/opus-mt-en-iso',
                     ('en', 'it'): 'Helsinki-NLP/opus-mt-en-it', ('en', 'itc'): 'Helsinki-NLP/opus-mt-en-itc',
                     ('en', 'jap'): 'Helsinki-NLP/opus-mt-en-jap', ('en', 'kg'): 'Helsinki-NLP/opus-mt-en-kg',
                     ('en', 'kj'): 'Helsinki-NLP/opus-mt-en-kj', ('en', 'kqn'): 'Helsinki-NLP/opus-mt-en-kqn',
                     ('en', 'kwn'): 'Helsinki-NLP/opus-mt-en-kwn', ('en', 'kwy'): 'Helsinki-NLP/opus-mt-en-kwy',
                     ('en', 'lg'): 'Helsinki-NLP/opus-mt-en-lg', ('en', 'ln'): 'Helsinki-NLP/opus-mt-en-ln',
                     ('en', 'loz'): 'Helsinki-NLP/opus-mt-en-loz', ('en', 'lu'): 'Helsinki-NLP/opus-mt-en-lu',
                     ('en', 'lua'): 'Helsinki-NLP/opus-mt-en-lua', ('en', 'lue'): 'Helsinki-NLP/opus-mt-en-lue',
                     ('en', 'lun'): 'Helsinki-NLP/opus-mt-en-lun', ('en', 'luo'): 'Helsinki-NLP/opus-mt-en-luo',
                     ('en', 'lus'): 'Helsinki-NLP/opus-mt-en-lus', ('en', 'map'): 'Helsinki-NLP/opus-mt-en-map',
                     ('en', 'mfe'): 'Helsinki-NLP/opus-mt-en-mfe', ('en', 'mg'): 'Helsinki-NLP/opus-mt-en-mg',
                     ('en', 'mh'): 'Helsinki-NLP/opus-mt-en-mh', ('en', 'mk'): 'Helsinki-NLP/opus-mt-en-mk',
                     ('en', 'mkh'): 'Helsinki-NLP/opus-mt-en-mkh', ('en', 'ml'): 'Helsinki-NLP/opus-mt-en-ml',
                     ('en', 'mos'): 'Helsinki-NLP/opus-mt-en-mos', ('en', 'mr'): 'Helsinki-NLP/opus-mt-en-mr',
                     ('en', 'mt'): 'Helsinki-NLP/opus-mt-en-mt', ('en', 'mul'): 'Helsinki-NLP/opus-mt-en-mul',
                     ('en', 'ng'): 'Helsinki-NLP/opus-mt-en-ng', ('en', 'nic'): 'Helsinki-NLP/opus-mt-en-nic',
                     ('en', 'niu'): 'Helsinki-NLP/opus-mt-en-niu', ('en', 'nl'): 'Helsinki-NLP/opus-mt-en-nl',
                     ('en', 'nso'): 'Helsinki-NLP/opus-mt-en-nso', ('en', 'ny'): 'Helsinki-NLP/opus-mt-en-ny',
                     ('en', 'nyk'): 'Helsinki-NLP/opus-mt-en-nyk', ('en', 'om'): 'Helsinki-NLP/opus-mt-en-om',
                     ('en', 'pag'): 'Helsinki-NLP/opus-mt-en-pag', ('en', 'pap'): 'Helsinki-NLP/opus-mt-en-pap',
                     ('en', 'phi'): 'Helsinki-NLP/opus-mt-en-phi', ('en', 'pis'): 'Helsinki-NLP/opus-mt-en-pis',
                     ('en', 'pon'): 'Helsinki-NLP/opus-mt-en-pon', ('en', 'poz'): 'Helsinki-NLP/opus-mt-en-poz',
                     ('en', 'pqe'): 'Helsinki-NLP/opus-mt-en-pqe', ('en', 'pqw'): 'Helsinki-NLP/opus-mt-en-pqw',
                     ('en', 'rn'): 'Helsinki-NLP/opus-mt-en-rn', ('en', 'rnd'): 'Helsinki-NLP/opus-mt-en-rnd',
                     ('en', 'ro'): 'Helsinki-NLP/opus-mt-en-ro', ('en', 'roa'): 'Helsinki-NLP/opus-mt-en-roa',
                     ('en', 'ru'): 'Helsinki-NLP/opus-mt-en-ru', ('en', 'run'): 'Helsinki-NLP/opus-mt-en-run',
                     ('en', 'rw'): 'Helsinki-NLP/opus-mt-en-rw', ('en', 'sal'): 'Helsinki-NLP/opus-mt-en-sal',
                     ('en', 'sem'): 'Helsinki-NLP/opus-mt-en-sem', ('en', 'sg'): 'Helsinki-NLP/opus-mt-en-sg',
                     ('en', 'sit'): 'Helsinki-NLP/opus-mt-en-sit', ('en', 'sk'): 'Helsinki-NLP/opus-mt-en-sk',
                     ('en', 'sla'): 'Helsinki-NLP/opus-mt-en-sla', ('en', 'sm'): 'Helsinki-NLP/opus-mt-en-sm',
                     ('en', 'sn'): 'Helsinki-NLP/opus-mt-en-sn', ('en', 'sq'): 'Helsinki-NLP/opus-mt-en-sq',
                     ('en', 'ss'): 'Helsinki-NLP/opus-mt-en-ss', ('en', 'st'): 'Helsinki-NLP/opus-mt-en-st',
                     ('en', 'sv'): 'Helsinki-NLP/opus-mt-en-sv', ('en', 'sw'): 'Helsinki-NLP/opus-mt-en-sw',
                     ('en', 'swc'): 'Helsinki-NLP/opus-mt-en-swc', ('en', 'tdt'): 'Helsinki-NLP/opus-mt-en-tdt',
                     ('en', 'ti'): 'Helsinki-NLP/opus-mt-en-ti', ('en', 'tiv'): 'Helsinki-NLP/opus-mt-en-tiv',
                     ('en', 'tl'): 'Helsinki-NLP/opus-mt-en-tl', ('en', 'tll'): 'Helsinki-NLP/opus-mt-en-tll',
                     ('en', 'tn'): 'Helsinki-NLP/opus-mt-en-tn', ('en', 'to'): 'Helsinki-NLP/opus-mt-en-to',
                     ('en', 'toi'): 'Helsinki-NLP/opus-mt-en-toi', ('en', 'tpi'): 'Helsinki-NLP/opus-mt-en-tpi',
                     ('en', 'trk'): 'Helsinki-NLP/opus-mt-en-trk', ('en', 'ts'): 'Helsinki-NLP/opus-mt-en-ts',
                     ('en', 'tut'): 'Helsinki-NLP/opus-mt-en-tut', ('en', 'tvl'): 'Helsinki-NLP/opus-mt-en-tvl',
                     ('en', 'tw'): 'Helsinki-NLP/opus-mt-en-tw', ('en', 'ty'): 'Helsinki-NLP/opus-mt-en-ty',
                     ('en', 'uk'): 'Helsinki-NLP/opus-mt-en-uk', ('en', 'umb'): 'Helsinki-NLP/opus-mt-en-umb',
                     ('en', 'ur'): 'Helsinki-NLP/opus-mt-en-ur', ('en', 'urj'): 'Helsinki-NLP/opus-mt-en-urj',
                     ('en', 'vi'): 'Helsinki-NLP/opus-mt-en-vi', ('en', 'xh'): 'Helsinki-NLP/opus-mt-en-xh',
                     ('en', 'zh'): 'Helsinki-NLP/opus-mt-en-zh', ('en', 'zle'): 'Helsinki-NLP/opus-mt-en-zle',
                     ('en', 'zls'): 'Helsinki-NLP/opus-mt-en-zls', ('en', 'zlw'): 'Helsinki-NLP/opus-mt-en-zlw',
                     ('en_el_es_fi', 'en_el_es_fi'): 'Helsinki-NLP/opus-mt-en_el_es_fi-en_el_es_fi',
                     ('eo', 'af'): 'Helsinki-NLP/opus-mt-eo-af', ('eo', 'bg'): 'Helsinki-NLP/opus-mt-eo-bg',
                     ('eo', 'cs'): 'Helsinki-NLP/opus-mt-eo-cs', ('eo', 'da'): 'Helsinki-NLP/opus-mt-eo-da',
                     ('eo', 'de'): 'Helsinki-NLP/opus-mt-eo-de', ('eo', 'el'): 'Helsinki-NLP/opus-mt-eo-el',
                     ('eo', 'en'): 'Helsinki-NLP/opus-mt-eo-en', ('eo', 'es'): 'Helsinki-NLP/opus-mt-eo-es',
                     ('eo', 'fi'): 'Helsinki-NLP/opus-mt-eo-fi', ('eo', 'fr'): 'Helsinki-NLP/opus-mt-eo-fr',
                     ('eo', 'he'): 'Helsinki-NLP/opus-mt-eo-he', ('eo', 'hu'): 'Helsinki-NLP/opus-mt-eo-hu',
                     ('eo', 'it'): 'Helsinki-NLP/opus-mt-eo-it', ('eo', 'nl'): 'Helsinki-NLP/opus-mt-eo-nl',
                     ('eo', 'pl'): 'Helsinki-NLP/opus-mt-eo-pl', ('eo', 'pt'): 'Helsinki-NLP/opus-mt-eo-pt',
                     ('eo', 'ro'): 'Helsinki-NLP/opus-mt-eo-ro', ('eo', 'ru'): 'Helsinki-NLP/opus-mt-eo-ru',
                     ('eo', 'sh'): 'Helsinki-NLP/opus-mt-eo-sh', ('eo', 'sv'): 'Helsinki-NLP/opus-mt-eo-sv',
                     ('es', 'NORWAY'): 'Helsinki-NLP/opus-mt-es-NORWAY', ('es', 'aed'): 'Helsinki-NLP/opus-mt-es-aed',
                     ('es', 'af'): 'Helsinki-NLP/opus-mt-es-af', ('es', 'ar'): 'Helsinki-NLP/opus-mt-es-ar',
                     ('es', 'ase'): 'Helsinki-NLP/opus-mt-es-ase', ('es', 'bcl'): 'Helsinki-NLP/opus-mt-es-bcl',
                     ('es', 'ber'): 'Helsinki-NLP/opus-mt-es-ber', ('es', 'bg'): 'Helsinki-NLP/opus-mt-es-bg',
                     ('es', 'bi'): 'Helsinki-NLP/opus-mt-es-bi', ('es', 'bzs'): 'Helsinki-NLP/opus-mt-es-bzs',
                     ('es', 'ca'): 'Helsinki-NLP/opus-mt-es-ca', ('es', 'ceb'): 'Helsinki-NLP/opus-mt-es-ceb',
                     ('es', 'crs'): 'Helsinki-NLP/opus-mt-es-crs', ('es', 'cs'): 'Helsinki-NLP/opus-mt-es-cs',
                     ('es', 'csg'): 'Helsinki-NLP/opus-mt-es-csg', ('es', 'csn'): 'Helsinki-NLP/opus-mt-es-csn',
                     ('es', 'da'): 'Helsinki-NLP/opus-mt-es-da', ('es', 'de'): 'Helsinki-NLP/opus-mt-es-de',
                     ('es', 'ee'): 'Helsinki-NLP/opus-mt-es-ee', ('es', 'efi'): 'Helsinki-NLP/opus-mt-es-efi',
                     ('es', 'el'): 'Helsinki-NLP/opus-mt-es-el', ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
                     ('es', 'eo'): 'Helsinki-NLP/opus-mt-es-eo', ('es', 'es'): 'Helsinki-NLP/opus-mt-es-es',
                     ('es', 'et'): 'Helsinki-NLP/opus-mt-es-et', ('es', 'eu'): 'Helsinki-NLP/opus-mt-es-eu',
                     ('es', 'fi'): 'Helsinki-NLP/opus-mt-es-fi', ('es', 'fj'): 'Helsinki-NLP/opus-mt-es-fj',
                     ('es', 'fr'): 'Helsinki-NLP/opus-mt-es-fr', ('es', 'gaa'): 'Helsinki-NLP/opus-mt-es-gaa',
                     ('es', 'gil'): 'Helsinki-NLP/opus-mt-es-gil', ('es', 'gl'): 'Helsinki-NLP/opus-mt-es-gl',
                     ('es', 'guw'): 'Helsinki-NLP/opus-mt-es-guw', ('es', 'ha'): 'Helsinki-NLP/opus-mt-es-ha',
                     ('es', 'he'): 'Helsinki-NLP/opus-mt-es-he', ('es', 'hil'): 'Helsinki-NLP/opus-mt-es-hil',
                     ('es', 'ho'): 'Helsinki-NLP/opus-mt-es-ho', ('es', 'hr'): 'Helsinki-NLP/opus-mt-es-hr',
                     ('es', 'ht'): 'Helsinki-NLP/opus-mt-es-ht', ('es', 'id'): 'Helsinki-NLP/opus-mt-es-id',
                     ('es', 'ig'): 'Helsinki-NLP/opus-mt-es-ig', ('es', 'ilo'): 'Helsinki-NLP/opus-mt-es-ilo',
                     ('es', 'is'): 'Helsinki-NLP/opus-mt-es-is', ('es', 'iso'): 'Helsinki-NLP/opus-mt-es-iso',
                     ('es', 'it'): 'Helsinki-NLP/opus-mt-es-it', ('es', 'kg'): 'Helsinki-NLP/opus-mt-es-kg',
                     ('es', 'ln'): 'Helsinki-NLP/opus-mt-es-ln', ('es', 'loz'): 'Helsinki-NLP/opus-mt-es-loz',
                     ('es', 'lt'): 'Helsinki-NLP/opus-mt-es-lt', ('es', 'lua'): 'Helsinki-NLP/opus-mt-es-lua',
                     ('es', 'lus'): 'Helsinki-NLP/opus-mt-es-lus', ('es', 'mfs'): 'Helsinki-NLP/opus-mt-es-mfs',
                     ('es', 'mk'): 'Helsinki-NLP/opus-mt-es-mk', ('es', 'mt'): 'Helsinki-NLP/opus-mt-es-mt',
                     ('es', 'niu'): 'Helsinki-NLP/opus-mt-es-niu', ('es', 'nl'): 'Helsinki-NLP/opus-mt-es-nl',
                     ('es', 'no'): 'Helsinki-NLP/opus-mt-es-no', ('es', 'nso'): 'Helsinki-NLP/opus-mt-es-nso',
                     ('es', 'ny'): 'Helsinki-NLP/opus-mt-es-ny', ('es', 'pag'): 'Helsinki-NLP/opus-mt-es-pag',
                     ('es', 'pap'): 'Helsinki-NLP/opus-mt-es-pap', ('es', 'pis'): 'Helsinki-NLP/opus-mt-es-pis',
                     ('es', 'pl'): 'Helsinki-NLP/opus-mt-es-pl', ('es', 'pon'): 'Helsinki-NLP/opus-mt-es-pon',
                     ('es', 'prl'): 'Helsinki-NLP/opus-mt-es-prl', ('es', 'rn'): 'Helsinki-NLP/opus-mt-es-rn',
                     ('es', 'ro'): 'Helsinki-NLP/opus-mt-es-ro', ('es', 'ru'): 'Helsinki-NLP/opus-mt-es-ru',
                     ('es', 'rw'): 'Helsinki-NLP/opus-mt-es-rw', ('es', 'sg'): 'Helsinki-NLP/opus-mt-es-sg',
                     ('es', 'sl'): 'Helsinki-NLP/opus-mt-es-sl', ('es', 'sm'): 'Helsinki-NLP/opus-mt-es-sm',
                     ('es', 'sn'): 'Helsinki-NLP/opus-mt-es-sn', ('es', 'srn'): 'Helsinki-NLP/opus-mt-es-srn',
                     ('es', 'st'): 'Helsinki-NLP/opus-mt-es-st', ('es', 'swc'): 'Helsinki-NLP/opus-mt-es-swc',
                     ('es', 'tl'): 'Helsinki-NLP/opus-mt-es-tl', ('es', 'tll'): 'Helsinki-NLP/opus-mt-es-tll',
                     ('es', 'tn'): 'Helsinki-NLP/opus-mt-es-tn', ('es', 'to'): 'Helsinki-NLP/opus-mt-es-to',
                     ('es', 'tpi'): 'Helsinki-NLP/opus-mt-es-tpi', ('es', 'tvl'): 'Helsinki-NLP/opus-mt-es-tvl',
                     ('es', 'tw'): 'Helsinki-NLP/opus-mt-es-tw', ('es', 'ty'): 'Helsinki-NLP/opus-mt-es-ty',
                     ('es', 'tzo'): 'Helsinki-NLP/opus-mt-es-tzo', ('es', 'uk'): 'Helsinki-NLP/opus-mt-es-uk',
                     ('es', 've'): 'Helsinki-NLP/opus-mt-es-ve', ('es', 'vi'): 'Helsinki-NLP/opus-mt-es-vi',
                     ('es', 'war'): 'Helsinki-NLP/opus-mt-es-war', ('es', 'wls'): 'Helsinki-NLP/opus-mt-es-wls',
                     ('es', 'xh'): 'Helsinki-NLP/opus-mt-es-xh', ('es', 'yo'): 'Helsinki-NLP/opus-mt-es-yo',
                     ('es', 'yua'): 'Helsinki-NLP/opus-mt-es-yua', ('es', 'zai'): 'Helsinki-NLP/opus-mt-es-zai',
                     ('et', 'de'): 'Helsinki-NLP/opus-mt-et-de', ('et', 'en'): 'Helsinki-NLP/opus-mt-et-en',
                     ('et', 'es'): 'Helsinki-NLP/opus-mt-et-es', ('et', 'fi'): 'Helsinki-NLP/opus-mt-et-fi',
                     ('et', 'fr'): 'Helsinki-NLP/opus-mt-et-fr', ('et', 'ru'): 'Helsinki-NLP/opus-mt-et-ru',
                     ('et', 'sv'): 'Helsinki-NLP/opus-mt-et-sv', ('eu', 'de'): 'Helsinki-NLP/opus-mt-eu-de',
                     ('eu', 'en'): 'Helsinki-NLP/opus-mt-eu-en', ('eu', 'es'): 'Helsinki-NLP/opus-mt-eu-es',
                     ('eu', 'ru'): 'Helsinki-NLP/opus-mt-eu-ru', ('euq', 'en'): 'Helsinki-NLP/opus-mt-euq-en',
                     ('fi', 'NORWAY'): 'Helsinki-NLP/opus-mt-fi-NORWAY', ('fi', 'ZH'): 'Helsinki-NLP/opus-mt-fi-ZH',
                     ('fi', 'af'): 'Helsinki-NLP/opus-mt-fi-af', ('fi', 'bcl'): 'Helsinki-NLP/opus-mt-fi-bcl',
                     ('fi', 'bem'): 'Helsinki-NLP/opus-mt-fi-bem', ('fi', 'bg'): 'Helsinki-NLP/opus-mt-fi-bg',
                     ('fi', 'bzs'): 'Helsinki-NLP/opus-mt-fi-bzs', ('fi', 'ceb'): 'Helsinki-NLP/opus-mt-fi-ceb',
                     ('fi', 'crs'): 'Helsinki-NLP/opus-mt-fi-crs', ('fi', 'cs'): 'Helsinki-NLP/opus-mt-fi-cs',
                     ('fi', 'de'): 'Helsinki-NLP/opus-mt-fi-de', ('fi', 'ee'): 'Helsinki-NLP/opus-mt-fi-ee',
                     ('fi', 'efi'): 'Helsinki-NLP/opus-mt-fi-efi', ('fi', 'el'): 'Helsinki-NLP/opus-mt-fi-el',
                     ('fi', 'en'): 'Helsinki-NLP/opus-mt-fi-en', ('fi', 'eo'): 'Helsinki-NLP/opus-mt-fi-eo',
                     ('fi', 'es'): 'Helsinki-NLP/opus-mt-fi-es', ('fi', 'et'): 'Helsinki-NLP/opus-mt-fi-et',
                     ('fi', 'fi'): 'Helsinki-NLP/opus-mt-fi-fi', ('fi', 'fj'): 'Helsinki-NLP/opus-mt-fi-fj',
                     ('fi', 'fr'): 'Helsinki-NLP/opus-mt-fi-fr', ('fi', 'fse'): 'Helsinki-NLP/opus-mt-fi-fse',
                     ('fi', 'gaa'): 'Helsinki-NLP/opus-mt-fi-gaa', ('fi', 'gil'): 'Helsinki-NLP/opus-mt-fi-gil',
                     ('fi', 'guw'): 'Helsinki-NLP/opus-mt-fi-guw', ('fi', 'ha'): 'Helsinki-NLP/opus-mt-fi-ha',
                     ('fi', 'he'): 'Helsinki-NLP/opus-mt-fi-he', ('fi', 'hil'): 'Helsinki-NLP/opus-mt-fi-hil',
                     ('fi', 'ho'): 'Helsinki-NLP/opus-mt-fi-ho', ('fi', 'hr'): 'Helsinki-NLP/opus-mt-fi-hr',
                     ('fi', 'ht'): 'Helsinki-NLP/opus-mt-fi-ht', ('fi', 'hu'): 'Helsinki-NLP/opus-mt-fi-hu',
                     ('fi', 'id'): 'Helsinki-NLP/opus-mt-fi-id', ('fi', 'ig'): 'Helsinki-NLP/opus-mt-fi-ig',
                     ('fi', 'ilo'): 'Helsinki-NLP/opus-mt-fi-ilo', ('fi', 'is'): 'Helsinki-NLP/opus-mt-fi-is',
                     ('fi', 'iso'): 'Helsinki-NLP/opus-mt-fi-iso', ('fi', 'it'): 'Helsinki-NLP/opus-mt-fi-it',
                     ('fi', 'kg'): 'Helsinki-NLP/opus-mt-fi-kg', ('fi', 'kqn'): 'Helsinki-NLP/opus-mt-fi-kqn',
                     ('fi', 'lg'): 'Helsinki-NLP/opus-mt-fi-lg', ('fi', 'ln'): 'Helsinki-NLP/opus-mt-fi-ln',
                     ('fi', 'lu'): 'Helsinki-NLP/opus-mt-fi-lu', ('fi', 'lua'): 'Helsinki-NLP/opus-mt-fi-lua',
                     ('fi', 'lue'): 'Helsinki-NLP/opus-mt-fi-lue', ('fi', 'lus'): 'Helsinki-NLP/opus-mt-fi-lus',
                     ('fi', 'lv'): 'Helsinki-NLP/opus-mt-fi-lv', ('fi', 'mfe'): 'Helsinki-NLP/opus-mt-fi-mfe',
                     ('fi', 'mg'): 'Helsinki-NLP/opus-mt-fi-mg', ('fi', 'mh'): 'Helsinki-NLP/opus-mt-fi-mh',
                     ('fi', 'mk'): 'Helsinki-NLP/opus-mt-fi-mk', ('fi', 'mos'): 'Helsinki-NLP/opus-mt-fi-mos',
                     ('fi', 'mt'): 'Helsinki-NLP/opus-mt-fi-mt', ('fi', 'niu'): 'Helsinki-NLP/opus-mt-fi-niu',
                     ('fi', 'nl'): 'Helsinki-NLP/opus-mt-fi-nl', ('fi', 'no'): 'Helsinki-NLP/opus-mt-fi-no',
                     ('fi', 'nso'): 'Helsinki-NLP/opus-mt-fi-nso', ('fi', 'ny'): 'Helsinki-NLP/opus-mt-fi-ny',
                     ('fi', 'pag'): 'Helsinki-NLP/opus-mt-fi-pag', ('fi', 'pap'): 'Helsinki-NLP/opus-mt-fi-pap',
                     ('fi', 'pis'): 'Helsinki-NLP/opus-mt-fi-pis', ('fi', 'pon'): 'Helsinki-NLP/opus-mt-fi-pon',
                     ('fi', 'ro'): 'Helsinki-NLP/opus-mt-fi-ro', ('fi', 'ru'): 'Helsinki-NLP/opus-mt-fi-ru',
                     ('fi', 'run'): 'Helsinki-NLP/opus-mt-fi-run', ('fi', 'rw'): 'Helsinki-NLP/opus-mt-fi-rw',
                     ('fi', 'sg'): 'Helsinki-NLP/opus-mt-fi-sg', ('fi', 'sk'): 'Helsinki-NLP/opus-mt-fi-sk',
                     ('fi', 'sl'): 'Helsinki-NLP/opus-mt-fi-sl', ('fi', 'sm'): 'Helsinki-NLP/opus-mt-fi-sm',
                     ('fi', 'sn'): 'Helsinki-NLP/opus-mt-fi-sn', ('fi', 'sq'): 'Helsinki-NLP/opus-mt-fi-sq',
                     ('fi', 'srn'): 'Helsinki-NLP/opus-mt-fi-srn', ('fi', 'st'): 'Helsinki-NLP/opus-mt-fi-st',
                     ('fi', 'sv'): 'Helsinki-NLP/opus-mt-fi-sv', ('fi', 'sw'): 'Helsinki-NLP/opus-mt-fi-sw',
                     ('fi', 'swc'): 'Helsinki-NLP/opus-mt-fi-swc', ('fi', 'tiv'): 'Helsinki-NLP/opus-mt-fi-tiv',
                     ('fi', 'tll'): 'Helsinki-NLP/opus-mt-fi-tll', ('fi', 'tn'): 'Helsinki-NLP/opus-mt-fi-tn',
                     ('fi', 'to'): 'Helsinki-NLP/opus-mt-fi-to', ('fi', 'toi'): 'Helsinki-NLP/opus-mt-fi-toi',
                     ('fi', 'tpi'): 'Helsinki-NLP/opus-mt-fi-tpi', ('fi', 'tr'): 'Helsinki-NLP/opus-mt-fi-tr',
                     ('fi', 'ts'): 'Helsinki-NLP/opus-mt-fi-ts', ('fi', 'tvl'): 'Helsinki-NLP/opus-mt-fi-tvl',
                     ('fi', 'tw'): 'Helsinki-NLP/opus-mt-fi-tw', ('fi', 'ty'): 'Helsinki-NLP/opus-mt-fi-ty',
                     ('fi', 'uk'): 'Helsinki-NLP/opus-mt-fi-uk', ('fi', 've'): 'Helsinki-NLP/opus-mt-fi-ve',
                     ('fi', 'war'): 'Helsinki-NLP/opus-mt-fi-war', ('fi', 'wls'): 'Helsinki-NLP/opus-mt-fi-wls',
                     ('fi', 'xh'): 'Helsinki-NLP/opus-mt-fi-xh', ('fi', 'yap'): 'Helsinki-NLP/opus-mt-fi-yap',
                     ('fi', 'yo'): 'Helsinki-NLP/opus-mt-fi-yo', ('fi', 'zne'): 'Helsinki-NLP/opus-mt-fi-zne',
                     ('fi_nb_no_nn_ru_sv_en', 'SAMI'): 'Helsinki-NLP/opus-mt-fi_nb_no_nn_ru_sv_en-SAMI',
                     ('fiu', 'en'): 'Helsinki-NLP/opus-mt-fiu-en', ('fiu', 'fiu'): 'Helsinki-NLP/opus-mt-fiu-fiu',
                     ('fj', 'en'): 'Helsinki-NLP/opus-mt-fj-en', ('fj', 'fr'): 'Helsinki-NLP/opus-mt-fj-fr',
                     ('fr', 'af'): 'Helsinki-NLP/opus-mt-fr-af', ('fr', 'ar'): 'Helsinki-NLP/opus-mt-fr-ar',
                     ('fr', 'ase'): 'Helsinki-NLP/opus-mt-fr-ase', ('fr', 'bcl'): 'Helsinki-NLP/opus-mt-fr-bcl',
                     ('fr', 'bem'): 'Helsinki-NLP/opus-mt-fr-bem', ('fr', 'ber'): 'Helsinki-NLP/opus-mt-fr-ber',
                     ('fr', 'bg'): 'Helsinki-NLP/opus-mt-fr-bg', ('fr', 'bi'): 'Helsinki-NLP/opus-mt-fr-bi',
                     ('fr', 'bzs'): 'Helsinki-NLP/opus-mt-fr-bzs', ('fr', 'ca'): 'Helsinki-NLP/opus-mt-fr-ca',
                     ('fr', 'ceb'): 'Helsinki-NLP/opus-mt-fr-ceb', ('fr', 'crs'): 'Helsinki-NLP/opus-mt-fr-crs',
                     ('fr', 'de'): 'Helsinki-NLP/opus-mt-fr-de', ('fr', 'ee'): 'Helsinki-NLP/opus-mt-fr-ee',
                     ('fr', 'efi'): 'Helsinki-NLP/opus-mt-fr-efi', ('fr', 'el'): 'Helsinki-NLP/opus-mt-fr-el',
                     ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en', ('fr', 'eo'): 'Helsinki-NLP/opus-mt-fr-eo',
                     ('fr', 'es'): 'Helsinki-NLP/opus-mt-fr-es', ('fr', 'fj'): 'Helsinki-NLP/opus-mt-fr-fj',
                     ('fr', 'gaa'): 'Helsinki-NLP/opus-mt-fr-gaa', ('fr', 'gil'): 'Helsinki-NLP/opus-mt-fr-gil',
                     ('fr', 'guw'): 'Helsinki-NLP/opus-mt-fr-guw', ('fr', 'ha'): 'Helsinki-NLP/opus-mt-fr-ha',
                     ('fr', 'he'): 'Helsinki-NLP/opus-mt-fr-he', ('fr', 'hil'): 'Helsinki-NLP/opus-mt-fr-hil',
                     ('fr', 'ho'): 'Helsinki-NLP/opus-mt-fr-ho', ('fr', 'hr'): 'Helsinki-NLP/opus-mt-fr-hr',
                     ('fr', 'ht'): 'Helsinki-NLP/opus-mt-fr-ht', ('fr', 'hu'): 'Helsinki-NLP/opus-mt-fr-hu',
                     ('fr', 'id'): 'Helsinki-NLP/opus-mt-fr-id', ('fr', 'ig'): 'Helsinki-NLP/opus-mt-fr-ig',
                     ('fr', 'ilo'): 'Helsinki-NLP/opus-mt-fr-ilo', ('fr', 'iso'): 'Helsinki-NLP/opus-mt-fr-iso',
                     ('fr', 'kg'): 'Helsinki-NLP/opus-mt-fr-kg', ('fr', 'kqn'): 'Helsinki-NLP/opus-mt-fr-kqn',
                     ('fr', 'kwy'): 'Helsinki-NLP/opus-mt-fr-kwy', ('fr', 'lg'): 'Helsinki-NLP/opus-mt-fr-lg',
                     ('fr', 'ln'): 'Helsinki-NLP/opus-mt-fr-ln', ('fr', 'loz'): 'Helsinki-NLP/opus-mt-fr-loz',
                     ('fr', 'lu'): 'Helsinki-NLP/opus-mt-fr-lu', ('fr', 'lua'): 'Helsinki-NLP/opus-mt-fr-lua',
                     ('fr', 'lue'): 'Helsinki-NLP/opus-mt-fr-lue', ('fr', 'lus'): 'Helsinki-NLP/opus-mt-fr-lus',
                     ('fr', 'mfe'): 'Helsinki-NLP/opus-mt-fr-mfe', ('fr', 'mh'): 'Helsinki-NLP/opus-mt-fr-mh',
                     ('fr', 'mos'): 'Helsinki-NLP/opus-mt-fr-mos', ('fr', 'ms'): 'Helsinki-NLP/opus-mt-fr-ms',
                     ('fr', 'mt'): 'Helsinki-NLP/opus-mt-fr-mt', ('fr', 'niu'): 'Helsinki-NLP/opus-mt-fr-niu',
                     ('fr', 'no'): 'Helsinki-NLP/opus-mt-fr-no', ('fr', 'nso'): 'Helsinki-NLP/opus-mt-fr-nso',
                     ('fr', 'ny'): 'Helsinki-NLP/opus-mt-fr-ny', ('fr', 'pag'): 'Helsinki-NLP/opus-mt-fr-pag',
                     ('fr', 'pap'): 'Helsinki-NLP/opus-mt-fr-pap', ('fr', 'pis'): 'Helsinki-NLP/opus-mt-fr-pis',
                     ('fr', 'pl'): 'Helsinki-NLP/opus-mt-fr-pl', ('fr', 'pon'): 'Helsinki-NLP/opus-mt-fr-pon',
                     ('fr', 'rnd'): 'Helsinki-NLP/opus-mt-fr-rnd', ('fr', 'ro'): 'Helsinki-NLP/opus-mt-fr-ro',
                     ('fr', 'ru'): 'Helsinki-NLP/opus-mt-fr-ru', ('fr', 'run'): 'Helsinki-NLP/opus-mt-fr-run',
                     ('fr', 'rw'): 'Helsinki-NLP/opus-mt-fr-rw', ('fr', 'sg'): 'Helsinki-NLP/opus-mt-fr-sg',
                     ('fr', 'sk'): 'Helsinki-NLP/opus-mt-fr-sk', ('fr', 'sl'): 'Helsinki-NLP/opus-mt-fr-sl',
                     ('fr', 'sm'): 'Helsinki-NLP/opus-mt-fr-sm', ('fr', 'sn'): 'Helsinki-NLP/opus-mt-fr-sn',
                     ('fr', 'srn'): 'Helsinki-NLP/opus-mt-fr-srn', ('fr', 'st'): 'Helsinki-NLP/opus-mt-fr-st',
                     ('fr', 'sv'): 'Helsinki-NLP/opus-mt-fr-sv', ('fr', 'swc'): 'Helsinki-NLP/opus-mt-fr-swc',
                     ('fr', 'tiv'): 'Helsinki-NLP/opus-mt-fr-tiv', ('fr', 'tl'): 'Helsinki-NLP/opus-mt-fr-tl',
                     ('fr', 'tll'): 'Helsinki-NLP/opus-mt-fr-tll', ('fr', 'tn'): 'Helsinki-NLP/opus-mt-fr-tn',
                     ('fr', 'to'): 'Helsinki-NLP/opus-mt-fr-to', ('fr', 'tpi'): 'Helsinki-NLP/opus-mt-fr-tpi',
                     ('fr', 'ts'): 'Helsinki-NLP/opus-mt-fr-ts', ('fr', 'tum'): 'Helsinki-NLP/opus-mt-fr-tum',
                     ('fr', 'tvl'): 'Helsinki-NLP/opus-mt-fr-tvl', ('fr', 'tw'): 'Helsinki-NLP/opus-mt-fr-tw',
                     ('fr', 'ty'): 'Helsinki-NLP/opus-mt-fr-ty', ('fr', 'uk'): 'Helsinki-NLP/opus-mt-fr-uk',
                     ('fr', 've'): 'Helsinki-NLP/opus-mt-fr-ve', ('fr', 'vi'): 'Helsinki-NLP/opus-mt-fr-vi',
                     ('fr', 'war'): 'Helsinki-NLP/opus-mt-fr-war', ('fr', 'wls'): 'Helsinki-NLP/opus-mt-fr-wls',
                     ('fr', 'xh'): 'Helsinki-NLP/opus-mt-fr-xh', ('fr', 'yap'): 'Helsinki-NLP/opus-mt-fr-yap',
                     ('fr', 'yo'): 'Helsinki-NLP/opus-mt-fr-yo', ('fr', 'zne'): 'Helsinki-NLP/opus-mt-fr-zne',
                     ('fse', 'fi'): 'Helsinki-NLP/opus-mt-fse-fi', ('ga', 'en'): 'Helsinki-NLP/opus-mt-ga-en',
                     ('gaa', 'de'): 'Helsinki-NLP/opus-mt-gaa-de', ('gaa', 'en'): 'Helsinki-NLP/opus-mt-gaa-en',
                     ('gaa', 'es'): 'Helsinki-NLP/opus-mt-gaa-es', ('gaa', 'fi'): 'Helsinki-NLP/opus-mt-gaa-fi',
                     ('gaa', 'fr'): 'Helsinki-NLP/opus-mt-gaa-fr', ('gaa', 'sv'): 'Helsinki-NLP/opus-mt-gaa-sv',
                     ('gem', 'en'): 'Helsinki-NLP/opus-mt-gem-en', ('gem', 'gem'): 'Helsinki-NLP/opus-mt-gem-gem',
                     ('gil', 'en'): 'Helsinki-NLP/opus-mt-gil-en', ('gil', 'es'): 'Helsinki-NLP/opus-mt-gil-es',
                     ('gil', 'fi'): 'Helsinki-NLP/opus-mt-gil-fi', ('gil', 'fr'): 'Helsinki-NLP/opus-mt-gil-fr',
                     ('gil', 'sv'): 'Helsinki-NLP/opus-mt-gil-sv', ('gl', 'en'): 'Helsinki-NLP/opus-mt-gl-en',
                     ('gl', 'es'): 'Helsinki-NLP/opus-mt-gl-es', ('gl', 'pt'): 'Helsinki-NLP/opus-mt-gl-pt',
                     ('gmq', 'en'): 'Helsinki-NLP/opus-mt-gmq-en', ('gmq', 'gmq'): 'Helsinki-NLP/opus-mt-gmq-gmq',
                     ('gmw', 'en'): 'Helsinki-NLP/opus-mt-gmw-en', ('gmw', 'gmw'): 'Helsinki-NLP/opus-mt-gmw-gmw',
                     ('grk', 'en'): 'Helsinki-NLP/opus-mt-grk-en', ('guw', 'de'): 'Helsinki-NLP/opus-mt-guw-de',
                     ('guw', 'en'): 'Helsinki-NLP/opus-mt-guw-en', ('guw', 'es'): 'Helsinki-NLP/opus-mt-guw-es',
                     ('guw', 'fi'): 'Helsinki-NLP/opus-mt-guw-fi', ('guw', 'fr'): 'Helsinki-NLP/opus-mt-guw-fr',
                     ('guw', 'sv'): 'Helsinki-NLP/opus-mt-guw-sv', ('gv', 'en'): 'Helsinki-NLP/opus-mt-gv-en',
                     ('ha', 'en'): 'Helsinki-NLP/opus-mt-ha-en', ('ha', 'es'): 'Helsinki-NLP/opus-mt-ha-es',
                     ('ha', 'fi'): 'Helsinki-NLP/opus-mt-ha-fi', ('ha', 'fr'): 'Helsinki-NLP/opus-mt-ha-fr',
                     ('ha', 'sv'): 'Helsinki-NLP/opus-mt-ha-sv', ('he', 'ar'): 'Helsinki-NLP/opus-mt-he-ar',
                     ('he', 'de'): 'Helsinki-NLP/opus-mt-he-de', ('he', 'eo'): 'Helsinki-NLP/opus-mt-he-eo',
                     ('he', 'es'): 'Helsinki-NLP/opus-mt-he-es', ('he', 'fi'): 'Helsinki-NLP/opus-mt-he-fi',
                     ('he', 'fr'): 'Helsinki-NLP/opus-mt-he-fr', ('he', 'it'): 'Helsinki-NLP/opus-mt-he-it',
                     ('he', 'ru'): 'Helsinki-NLP/opus-mt-he-ru', ('he', 'sv'): 'Helsinki-NLP/opus-mt-he-sv',
                     ('he', 'uk'): 'Helsinki-NLP/opus-mt-he-uk', ('hi', 'en'): 'Helsinki-NLP/opus-mt-hi-en',
                     ('hi', 'ur'): 'Helsinki-NLP/opus-mt-hi-ur', ('hil', 'de'): 'Helsinki-NLP/opus-mt-hil-de',
                     ('hil', 'en'): 'Helsinki-NLP/opus-mt-hil-en', ('hil', 'fi'): 'Helsinki-NLP/opus-mt-hil-fi',
                     ('ho', 'en'): 'Helsinki-NLP/opus-mt-ho-en', ('hr', 'es'): 'Helsinki-NLP/opus-mt-hr-es',
                     ('hr', 'fi'): 'Helsinki-NLP/opus-mt-hr-fi', ('hr', 'fr'): 'Helsinki-NLP/opus-mt-hr-fr',
                     ('hr', 'sv'): 'Helsinki-NLP/opus-mt-hr-sv', ('ht', 'en'): 'Helsinki-NLP/opus-mt-ht-en',
                     ('ht', 'es'): 'Helsinki-NLP/opus-mt-ht-es', ('ht', 'fi'): 'Helsinki-NLP/opus-mt-ht-fi',
                     ('ht', 'fr'): 'Helsinki-NLP/opus-mt-ht-fr', ('ht', 'sv'): 'Helsinki-NLP/opus-mt-ht-sv',
                     ('hu', 'de'): 'Helsinki-NLP/opus-mt-hu-de', ('hu', 'en'): 'Helsinki-NLP/opus-mt-hu-en',
                     ('hu', 'eo'): 'Helsinki-NLP/opus-mt-hu-eo', ('hu', 'fi'): 'Helsinki-NLP/opus-mt-hu-fi',
                     ('hu', 'fr'): 'Helsinki-NLP/opus-mt-hu-fr', ('hu', 'sv'): 'Helsinki-NLP/opus-mt-hu-sv',
                     ('hu', 'uk'): 'Helsinki-NLP/opus-mt-hu-uk', ('hy', 'en'): 'Helsinki-NLP/opus-mt-hy-en',
                     ('hy', 'ru'): 'Helsinki-NLP/opus-mt-hy-ru', ('id', 'en'): 'Helsinki-NLP/opus-mt-id-en',
                     ('id', 'es'): 'Helsinki-NLP/opus-mt-id-es', ('id', 'fi'): 'Helsinki-NLP/opus-mt-id-fi',
                     ('id', 'fr'): 'Helsinki-NLP/opus-mt-id-fr', ('id', 'sv'): 'Helsinki-NLP/opus-mt-id-sv',
                     ('ig', 'de'): 'Helsinki-NLP/opus-mt-ig-de', ('ig', 'en'): 'Helsinki-NLP/opus-mt-ig-en',
                     ('ig', 'es'): 'Helsinki-NLP/opus-mt-ig-es', ('ig', 'fi'): 'Helsinki-NLP/opus-mt-ig-fi',
                     ('ig', 'fr'): 'Helsinki-NLP/opus-mt-ig-fr', ('ig', 'sv'): 'Helsinki-NLP/opus-mt-ig-sv',
                     ('iir', 'en'): 'Helsinki-NLP/opus-mt-iir-en', ('iir', 'iir'): 'Helsinki-NLP/opus-mt-iir-iir',
                     ('ilo', 'de'): 'Helsinki-NLP/opus-mt-ilo-de', ('ilo', 'en'): 'Helsinki-NLP/opus-mt-ilo-en',
                     ('ilo', 'es'): 'Helsinki-NLP/opus-mt-ilo-es', ('ilo', 'fi'): 'Helsinki-NLP/opus-mt-ilo-fi',
                     ('ilo', 'sv'): 'Helsinki-NLP/opus-mt-ilo-sv', ('inc', 'en'): 'Helsinki-NLP/opus-mt-inc-en',
                     ('inc', 'inc'): 'Helsinki-NLP/opus-mt-inc-inc', ('ine', 'en'): 'Helsinki-NLP/opus-mt-ine-en',
                     ('ine', 'ine'): 'Helsinki-NLP/opus-mt-ine-ine', ('is', 'de'): 'Helsinki-NLP/opus-mt-is-de',
                     ('is', 'en'): 'Helsinki-NLP/opus-mt-is-en', ('is', 'eo'): 'Helsinki-NLP/opus-mt-is-eo',
                     ('is', 'es'): 'Helsinki-NLP/opus-mt-is-es', ('is', 'fi'): 'Helsinki-NLP/opus-mt-is-fi',
                     ('is', 'fr'): 'Helsinki-NLP/opus-mt-is-fr', ('is', 'it'): 'Helsinki-NLP/opus-mt-is-it',
                     ('is', 'sv'): 'Helsinki-NLP/opus-mt-is-sv', ('iso', 'en'): 'Helsinki-NLP/opus-mt-iso-en',
                     ('iso', 'es'): 'Helsinki-NLP/opus-mt-iso-es', ('iso', 'fi'): 'Helsinki-NLP/opus-mt-iso-fi',
                     ('iso', 'fr'): 'Helsinki-NLP/opus-mt-iso-fr', ('iso', 'sv'): 'Helsinki-NLP/opus-mt-iso-sv',
                     ('it', 'ar'): 'Helsinki-NLP/opus-mt-it-ar', ('it', 'bg'): 'Helsinki-NLP/opus-mt-it-bg',
                     ('it', 'ca'): 'Helsinki-NLP/opus-mt-it-ca', ('it', 'de'): 'Helsinki-NLP/opus-mt-it-de',
                     ('it', 'en'): 'Helsinki-NLP/opus-mt-it-en', ('it', 'eo'): 'Helsinki-NLP/opus-mt-it-eo',
                     ('it', 'es'): 'Helsinki-NLP/opus-mt-it-es', ('it', 'fr'): 'Helsinki-NLP/opus-mt-it-fr',
                     ('it', 'is'): 'Helsinki-NLP/opus-mt-it-is', ('it', 'lt'): 'Helsinki-NLP/opus-mt-it-lt',
                     ('it', 'ms'): 'Helsinki-NLP/opus-mt-it-ms', ('it', 'sv'): 'Helsinki-NLP/opus-mt-it-sv',
                     ('it', 'uk'): 'Helsinki-NLP/opus-mt-it-uk', ('it', 'vi'): 'Helsinki-NLP/opus-mt-it-vi',
                     ('itc', 'en'): 'Helsinki-NLP/opus-mt-itc-en', ('itc', 'itc'): 'Helsinki-NLP/opus-mt-itc-itc',
                     ('ja', 'ar'): 'Helsinki-NLP/opus-mt-ja-ar', ('ja', 'bg'): 'Helsinki-NLP/opus-mt-ja-bg',
                     ('ja', 'da'): 'Helsinki-NLP/opus-mt-ja-da', ('ja', 'de'): 'Helsinki-NLP/opus-mt-ja-de',
                     ('ja', 'en'): 'Helsinki-NLP/opus-mt-ja-en', ('ja', 'es'): 'Helsinki-NLP/opus-mt-ja-es',
                     ('ja', 'fi'): 'Helsinki-NLP/opus-mt-ja-fi', ('ja', 'fr'): 'Helsinki-NLP/opus-mt-ja-fr',
                     ('ja', 'he'): 'Helsinki-NLP/opus-mt-ja-he', ('ja', 'hu'): 'Helsinki-NLP/opus-mt-ja-hu',
                     ('ja', 'it'): 'Helsinki-NLP/opus-mt-ja-it', ('ja', 'ms'): 'Helsinki-NLP/opus-mt-ja-ms',
                     ('ja', 'nl'): 'Helsinki-NLP/opus-mt-ja-nl', ('ja', 'pl'): 'Helsinki-NLP/opus-mt-ja-pl',
                     ('ja', 'pt'): 'Helsinki-NLP/opus-mt-ja-pt', ('ja', 'ru'): 'Helsinki-NLP/opus-mt-ja-ru',
                     ('ja', 'sh'): 'Helsinki-NLP/opus-mt-ja-sh', ('ja', 'sv'): 'Helsinki-NLP/opus-mt-ja-sv',
                     ('ja', 'tr'): 'Helsinki-NLP/opus-mt-ja-tr', ('ja', 'vi'): 'Helsinki-NLP/opus-mt-ja-vi',
                     ('jap', 'en'): 'Helsinki-NLP/opus-mt-jap-en', ('ka', 'en'): 'Helsinki-NLP/opus-mt-ka-en',
                     ('ka', 'ru'): 'Helsinki-NLP/opus-mt-ka-ru', ('kab', 'en'): 'Helsinki-NLP/opus-mt-kab-en',
                     ('kg', 'en'): 'Helsinki-NLP/opus-mt-kg-en', ('kg', 'es'): 'Helsinki-NLP/opus-mt-kg-es',
                     ('kg', 'fr'): 'Helsinki-NLP/opus-mt-kg-fr', ('kg', 'sv'): 'Helsinki-NLP/opus-mt-kg-sv',
                     ('kj', 'en'): 'Helsinki-NLP/opus-mt-kj-en', ('kl', 'en'): 'Helsinki-NLP/opus-mt-kl-en',
                     ('ko', 'de'): 'Helsinki-NLP/opus-mt-ko-de', ('ko', 'en'): 'Helsinki-NLP/opus-mt-ko-en',
                     ('ko', 'es'): 'Helsinki-NLP/opus-mt-ko-es', ('ko', 'fi'): 'Helsinki-NLP/opus-mt-ko-fi',
                     ('ko', 'fr'): 'Helsinki-NLP/opus-mt-ko-fr', ('ko', 'hu'): 'Helsinki-NLP/opus-mt-ko-hu',
                     ('ko', 'ru'): 'Helsinki-NLP/opus-mt-ko-ru', ('ko', 'sv'): 'Helsinki-NLP/opus-mt-ko-sv',
                     ('kqn', 'en'): 'Helsinki-NLP/opus-mt-kqn-en', ('kqn', 'es'): 'Helsinki-NLP/opus-mt-kqn-es',
                     ('kqn', 'fr'): 'Helsinki-NLP/opus-mt-kqn-fr', ('kqn', 'sv'): 'Helsinki-NLP/opus-mt-kqn-sv',
                     ('kwn', 'en'): 'Helsinki-NLP/opus-mt-kwn-en', ('kwy', 'en'): 'Helsinki-NLP/opus-mt-kwy-en',
                     ('kwy', 'fr'): 'Helsinki-NLP/opus-mt-kwy-fr', ('kwy', 'sv'): 'Helsinki-NLP/opus-mt-kwy-sv',
                     ('lg', 'en'): 'Helsinki-NLP/opus-mt-lg-en', ('lg', 'es'): 'Helsinki-NLP/opus-mt-lg-es',
                     ('lg', 'fi'): 'Helsinki-NLP/opus-mt-lg-fi', ('lg', 'fr'): 'Helsinki-NLP/opus-mt-lg-fr',
                     ('lg', 'sv'): 'Helsinki-NLP/opus-mt-lg-sv', ('ln', 'de'): 'Helsinki-NLP/opus-mt-ln-de',
                     ('ln', 'en'): 'Helsinki-NLP/opus-mt-ln-en', ('ln', 'es'): 'Helsinki-NLP/opus-mt-ln-es',
                     ('ln', 'fr'): 'Helsinki-NLP/opus-mt-ln-fr', ('loz', 'de'): 'Helsinki-NLP/opus-mt-loz-de',
                     ('loz', 'en'): 'Helsinki-NLP/opus-mt-loz-en', ('loz', 'es'): 'Helsinki-NLP/opus-mt-loz-es',
                     ('loz', 'fi'): 'Helsinki-NLP/opus-mt-loz-fi', ('loz', 'fr'): 'Helsinki-NLP/opus-mt-loz-fr',
                     ('loz', 'sv'): 'Helsinki-NLP/opus-mt-loz-sv', ('lt', 'de'): 'Helsinki-NLP/opus-mt-lt-de',
                     ('lt', 'eo'): 'Helsinki-NLP/opus-mt-lt-eo', ('lt', 'es'): 'Helsinki-NLP/opus-mt-lt-es',
                     ('lt', 'fr'): 'Helsinki-NLP/opus-mt-lt-fr', ('lt', 'it'): 'Helsinki-NLP/opus-mt-lt-it',
                     ('lt', 'pl'): 'Helsinki-NLP/opus-mt-lt-pl', ('lt', 'ru'): 'Helsinki-NLP/opus-mt-lt-ru',
                     ('lt', 'sv'): 'Helsinki-NLP/opus-mt-lt-sv', ('lt', 'tr'): 'Helsinki-NLP/opus-mt-lt-tr',
                     ('lu', 'en'): 'Helsinki-NLP/opus-mt-lu-en', ('lu', 'es'): 'Helsinki-NLP/opus-mt-lu-es',
                     ('lu', 'fi'): 'Helsinki-NLP/opus-mt-lu-fi', ('lu', 'fr'): 'Helsinki-NLP/opus-mt-lu-fr',
                     ('lu', 'sv'): 'Helsinki-NLP/opus-mt-lu-sv', ('lua', 'en'): 'Helsinki-NLP/opus-mt-lua-en',
                     ('lua', 'es'): 'Helsinki-NLP/opus-mt-lua-es', ('lua', 'fi'): 'Helsinki-NLP/opus-mt-lua-fi',
                     ('lua', 'fr'): 'Helsinki-NLP/opus-mt-lua-fr', ('lua', 'sv'): 'Helsinki-NLP/opus-mt-lua-sv',
                     ('lue', 'en'): 'Helsinki-NLP/opus-mt-lue-en', ('lue', 'es'): 'Helsinki-NLP/opus-mt-lue-es',
                     ('lue', 'fi'): 'Helsinki-NLP/opus-mt-lue-fi', ('lue', 'fr'): 'Helsinki-NLP/opus-mt-lue-fr',
                     ('lue', 'sv'): 'Helsinki-NLP/opus-mt-lue-sv', ('lun', 'en'): 'Helsinki-NLP/opus-mt-lun-en',
                     ('luo', 'en'): 'Helsinki-NLP/opus-mt-luo-en', ('lus', 'en'): 'Helsinki-NLP/opus-mt-lus-en',
                     ('lus', 'es'): 'Helsinki-NLP/opus-mt-lus-es', ('lus', 'fi'): 'Helsinki-NLP/opus-mt-lus-fi',
                     ('lus', 'fr'): 'Helsinki-NLP/opus-mt-lus-fr', ('lus', 'sv'): 'Helsinki-NLP/opus-mt-lus-sv',
                     ('lv', 'en'): 'Helsinki-NLP/opus-mt-lv-en', ('lv', 'es'): 'Helsinki-NLP/opus-mt-lv-es',
                     ('lv', 'fi'): 'Helsinki-NLP/opus-mt-lv-fi', ('lv', 'fr'): 'Helsinki-NLP/opus-mt-lv-fr',
                     ('lv', 'ru'): 'Helsinki-NLP/opus-mt-lv-ru', ('lv', 'sv'): 'Helsinki-NLP/opus-mt-lv-sv',
                     ('mfe', 'en'): 'Helsinki-NLP/opus-mt-mfe-en', ('mfe', 'es'): 'Helsinki-NLP/opus-mt-mfe-es',
                     ('mfs', 'es'): 'Helsinki-NLP/opus-mt-mfs-es', ('mg', 'en'): 'Helsinki-NLP/opus-mt-mg-en',
                     ('mg', 'es'): 'Helsinki-NLP/opus-mt-mg-es', ('mh', 'en'): 'Helsinki-NLP/opus-mt-mh-en',
                     ('mh', 'es'): 'Helsinki-NLP/opus-mt-mh-es', ('mh', 'fi'): 'Helsinki-NLP/opus-mt-mh-fi',
                     ('mk', 'en'): 'Helsinki-NLP/opus-mt-mk-en', ('mk', 'es'): 'Helsinki-NLP/opus-mt-mk-es',
                     ('mk', 'fi'): 'Helsinki-NLP/opus-mt-mk-fi', ('mk', 'fr'): 'Helsinki-NLP/opus-mt-mk-fr',
                     ('mkh', 'en'): 'Helsinki-NLP/opus-mt-mkh-en', ('ml', 'en'): 'Helsinki-NLP/opus-mt-ml-en',
                     ('mos', 'en'): 'Helsinki-NLP/opus-mt-mos-en', ('mr', 'en'): 'Helsinki-NLP/opus-mt-mr-en',
                     ('ms', 'de'): 'Helsinki-NLP/opus-mt-ms-de', ('ms', 'fr'): 'Helsinki-NLP/opus-mt-ms-fr',
                     ('ms', 'it'): 'Helsinki-NLP/opus-mt-ms-it', ('ms', 'ms'): 'Helsinki-NLP/opus-mt-ms-ms',
                     ('mt', 'en'): 'Helsinki-NLP/opus-mt-mt-en', ('mt', 'es'): 'Helsinki-NLP/opus-mt-mt-es',
                     ('mt', 'fi'): 'Helsinki-NLP/opus-mt-mt-fi', ('mt', 'fr'): 'Helsinki-NLP/opus-mt-mt-fr',
                     ('mt', 'sv'): 'Helsinki-NLP/opus-mt-mt-sv', ('mul', 'en'): 'Helsinki-NLP/opus-mt-mul-en',
                     ('ng', 'en'): 'Helsinki-NLP/opus-mt-ng-en', ('nic', 'en'): 'Helsinki-NLP/opus-mt-nic-en',
                     ('niu', 'de'): 'Helsinki-NLP/opus-mt-niu-de', ('niu', 'en'): 'Helsinki-NLP/opus-mt-niu-en',
                     ('niu', 'es'): 'Helsinki-NLP/opus-mt-niu-es', ('niu', 'fi'): 'Helsinki-NLP/opus-mt-niu-fi',
                     ('niu', 'fr'): 'Helsinki-NLP/opus-mt-niu-fr', ('niu', 'sv'): 'Helsinki-NLP/opus-mt-niu-sv',
                     ('nl', 'af'): 'Helsinki-NLP/opus-mt-nl-af', ('nl', 'ca'): 'Helsinki-NLP/opus-mt-nl-ca',
                     ('nl', 'en'): 'Helsinki-NLP/opus-mt-nl-en', ('nl', 'eo'): 'Helsinki-NLP/opus-mt-nl-eo',
                     ('nl', 'es'): 'Helsinki-NLP/opus-mt-nl-es', ('nl', 'fi'): 'Helsinki-NLP/opus-mt-nl-fi',
                     ('nl', 'fr'): 'Helsinki-NLP/opus-mt-nl-fr', ('nl', 'no'): 'Helsinki-NLP/opus-mt-nl-no',
                     ('nl', 'sv'): 'Helsinki-NLP/opus-mt-nl-sv', ('nl', 'uk'): 'Helsinki-NLP/opus-mt-nl-uk',
                     ('no', 'da'): 'Helsinki-NLP/opus-mt-no-da', ('no', 'de'): 'Helsinki-NLP/opus-mt-no-de',
                     ('no', 'es'): 'Helsinki-NLP/opus-mt-no-es', ('no', 'fi'): 'Helsinki-NLP/opus-mt-no-fi',
                     ('no', 'fr'): 'Helsinki-NLP/opus-mt-no-fr', ('no', 'nl'): 'Helsinki-NLP/opus-mt-no-nl',
                     ('no', 'no'): 'Helsinki-NLP/opus-mt-no-no', ('no', 'pl'): 'Helsinki-NLP/opus-mt-no-pl',
                     ('no', 'ru'): 'Helsinki-NLP/opus-mt-no-ru', ('no', 'sv'): 'Helsinki-NLP/opus-mt-no-sv',
                     ('no', 'uk'): 'Helsinki-NLP/opus-mt-no-uk', ('nso', 'de'): 'Helsinki-NLP/opus-mt-nso-de',
                     ('nso', 'en'): 'Helsinki-NLP/opus-mt-nso-en', ('nso', 'es'): 'Helsinki-NLP/opus-mt-nso-es',
                     ('nso', 'fi'): 'Helsinki-NLP/opus-mt-nso-fi', ('nso', 'fr'): 'Helsinki-NLP/opus-mt-nso-fr',
                     ('nso', 'sv'): 'Helsinki-NLP/opus-mt-nso-sv', ('ny', 'de'): 'Helsinki-NLP/opus-mt-ny-de',
                     ('ny', 'en'): 'Helsinki-NLP/opus-mt-ny-en', ('ny', 'es'): 'Helsinki-NLP/opus-mt-ny-es',
                     ('nyk', 'en'): 'Helsinki-NLP/opus-mt-nyk-en', ('om', 'en'): 'Helsinki-NLP/opus-mt-om-en',
                     ('pa', 'en'): 'Helsinki-NLP/opus-mt-pa-en', ('pag', 'de'): 'Helsinki-NLP/opus-mt-pag-de',
                     ('pag', 'en'): 'Helsinki-NLP/opus-mt-pag-en', ('pag', 'es'): 'Helsinki-NLP/opus-mt-pag-es',
                     ('pag', 'fi'): 'Helsinki-NLP/opus-mt-pag-fi', ('pag', 'sv'): 'Helsinki-NLP/opus-mt-pag-sv',
                     ('pap', 'de'): 'Helsinki-NLP/opus-mt-pap-de', ('pap', 'en'): 'Helsinki-NLP/opus-mt-pap-en',
                     ('pap', 'es'): 'Helsinki-NLP/opus-mt-pap-es', ('pap', 'fi'): 'Helsinki-NLP/opus-mt-pap-fi',
                     ('pap', 'fr'): 'Helsinki-NLP/opus-mt-pap-fr', ('phi', 'en'): 'Helsinki-NLP/opus-mt-phi-en',
                     ('pis', 'en'): 'Helsinki-NLP/opus-mt-pis-en', ('pis', 'es'): 'Helsinki-NLP/opus-mt-pis-es',
                     ('pis', 'fi'): 'Helsinki-NLP/opus-mt-pis-fi', ('pis', 'fr'): 'Helsinki-NLP/opus-mt-pis-fr',
                     ('pis', 'sv'): 'Helsinki-NLP/opus-mt-pis-sv', ('pl', 'ar'): 'Helsinki-NLP/opus-mt-pl-ar',
                     ('pl', 'de'): 'Helsinki-NLP/opus-mt-pl-de', ('pl', 'en'): 'Helsinki-NLP/opus-mt-pl-en',
                     ('pl', 'eo'): 'Helsinki-NLP/opus-mt-pl-eo', ('pl', 'es'): 'Helsinki-NLP/opus-mt-pl-es',
                     ('pl', 'fr'): 'Helsinki-NLP/opus-mt-pl-fr', ('pl', 'lt'): 'Helsinki-NLP/opus-mt-pl-lt',
                     ('pl', 'no'): 'Helsinki-NLP/opus-mt-pl-no', ('pl', 'sv'): 'Helsinki-NLP/opus-mt-pl-sv',
                     ('pl', 'uk'): 'Helsinki-NLP/opus-mt-pl-uk', ('pon', 'en'): 'Helsinki-NLP/opus-mt-pon-en',
                     ('pon', 'es'): 'Helsinki-NLP/opus-mt-pon-es', ('pon', 'fi'): 'Helsinki-NLP/opus-mt-pon-fi',
                     ('pon', 'fr'): 'Helsinki-NLP/opus-mt-pon-fr', ('pon', 'sv'): 'Helsinki-NLP/opus-mt-pon-sv',
                     ('pqe', 'en'): 'Helsinki-NLP/opus-mt-pqe-en', ('prl', 'es'): 'Helsinki-NLP/opus-mt-prl-es',
                     ('pt', 'ca'): 'Helsinki-NLP/opus-mt-pt-ca', ('pt', 'eo'): 'Helsinki-NLP/opus-mt-pt-eo',
                     ('pt', 'gl'): 'Helsinki-NLP/opus-mt-pt-gl', ('pt', 'tl'): 'Helsinki-NLP/opus-mt-pt-tl',
                     ('pt', 'uk'): 'Helsinki-NLP/opus-mt-pt-uk', ('rn', 'de'): 'Helsinki-NLP/opus-mt-rn-de',
                     ('rn', 'en'): 'Helsinki-NLP/opus-mt-rn-en', ('rn', 'es'): 'Helsinki-NLP/opus-mt-rn-es',
                     ('rn', 'fr'): 'Helsinki-NLP/opus-mt-rn-fr', ('rn', 'ru'): 'Helsinki-NLP/opus-mt-rn-ru',
                     ('rnd', 'en'): 'Helsinki-NLP/opus-mt-rnd-en', ('rnd', 'fr'): 'Helsinki-NLP/opus-mt-rnd-fr',
                     ('rnd', 'sv'): 'Helsinki-NLP/opus-mt-rnd-sv', ('ro', 'eo'): 'Helsinki-NLP/opus-mt-ro-eo',
                     ('ro', 'fi'): 'Helsinki-NLP/opus-mt-ro-fi', ('ro', 'fr'): 'Helsinki-NLP/opus-mt-ro-fr',
                     ('ro', 'sv'): 'Helsinki-NLP/opus-mt-ro-sv', ('roa', 'en'): 'Helsinki-NLP/opus-mt-roa-en',
                     ('ru', 'af'): 'Helsinki-NLP/opus-mt-ru-af', ('ru', 'ar'): 'Helsinki-NLP/opus-mt-ru-ar',
                     ('ru', 'bg'): 'Helsinki-NLP/opus-mt-ru-bg', ('ru', 'da'): 'Helsinki-NLP/opus-mt-ru-da',
                     ('ru', 'en'): 'Helsinki-NLP/opus-mt-ru-en', ('ru', 'eo'): 'Helsinki-NLP/opus-mt-ru-eo',
                     ('ru', 'es'): 'Helsinki-NLP/opus-mt-ru-es', ('ru', 'et'): 'Helsinki-NLP/opus-mt-ru-et',
                     ('ru', 'eu'): 'Helsinki-NLP/opus-mt-ru-eu', ('ru', 'fi'): 'Helsinki-NLP/opus-mt-ru-fi',
                     ('ru', 'fr'): 'Helsinki-NLP/opus-mt-ru-fr', ('ru', 'he'): 'Helsinki-NLP/opus-mt-ru-he',
                     ('ru', 'hy'): 'Helsinki-NLP/opus-mt-ru-hy', ('ru', 'lt'): 'Helsinki-NLP/opus-mt-ru-lt',
                     ('ru', 'lv'): 'Helsinki-NLP/opus-mt-ru-lv', ('ru', 'no'): 'Helsinki-NLP/opus-mt-ru-no',
                     ('ru', 'sl'): 'Helsinki-NLP/opus-mt-ru-sl', ('ru', 'sv'): 'Helsinki-NLP/opus-mt-ru-sv',
                     ('ru', 'uk'): 'Helsinki-NLP/opus-mt-ru-uk', ('ru', 'vi'): 'Helsinki-NLP/opus-mt-ru-vi',
                     ('run', 'en'): 'Helsinki-NLP/opus-mt-run-en', ('run', 'es'): 'Helsinki-NLP/opus-mt-run-es',
                     ('run', 'sv'): 'Helsinki-NLP/opus-mt-run-sv', ('rw', 'en'): 'Helsinki-NLP/opus-mt-rw-en',
                     ('rw', 'es'): 'Helsinki-NLP/opus-mt-rw-es', ('rw', 'fr'): 'Helsinki-NLP/opus-mt-rw-fr',
                     ('rw', 'sv'): 'Helsinki-NLP/opus-mt-rw-sv', ('sal', 'en'): 'Helsinki-NLP/opus-mt-sal-en',
                     ('sem', 'en'): 'Helsinki-NLP/opus-mt-sem-en', ('sem', 'sem'): 'Helsinki-NLP/opus-mt-sem-sem',
                     ('sg', 'en'): 'Helsinki-NLP/opus-mt-sg-en', ('sg', 'es'): 'Helsinki-NLP/opus-mt-sg-es',
                     ('sg', 'fi'): 'Helsinki-NLP/opus-mt-sg-fi', ('sg', 'fr'): 'Helsinki-NLP/opus-mt-sg-fr',
                     ('sg', 'sv'): 'Helsinki-NLP/opus-mt-sg-sv', ('sh', 'eo'): 'Helsinki-NLP/opus-mt-sh-eo',
                     ('sh', 'uk'): 'Helsinki-NLP/opus-mt-sh-uk', ('sk', 'en'): 'Helsinki-NLP/opus-mt-sk-en',
                     ('sk', 'es'): 'Helsinki-NLP/opus-mt-sk-es', ('sk', 'fi'): 'Helsinki-NLP/opus-mt-sk-fi',
                     ('sk', 'fr'): 'Helsinki-NLP/opus-mt-sk-fr', ('sk', 'sv'): 'Helsinki-NLP/opus-mt-sk-sv',
                     ('sl', 'es'): 'Helsinki-NLP/opus-mt-sl-es', ('sl', 'fi'): 'Helsinki-NLP/opus-mt-sl-fi',
                     ('sl', 'fr'): 'Helsinki-NLP/opus-mt-sl-fr', ('sl', 'ru'): 'Helsinki-NLP/opus-mt-sl-ru',
                     ('sl', 'sv'): 'Helsinki-NLP/opus-mt-sl-sv', ('sl', 'uk'): 'Helsinki-NLP/opus-mt-sl-uk',
                     ('sla', 'en'): 'Helsinki-NLP/opus-mt-sla-en', ('sla', 'sla'): 'Helsinki-NLP/opus-mt-sla-sla',
                     ('sm', 'en'): 'Helsinki-NLP/opus-mt-sm-en', ('sm', 'es'): 'Helsinki-NLP/opus-mt-sm-es',
                     ('sm', 'fr'): 'Helsinki-NLP/opus-mt-sm-fr', ('sn', 'en'): 'Helsinki-NLP/opus-mt-sn-en',
                     ('sn', 'es'): 'Helsinki-NLP/opus-mt-sn-es', ('sn', 'fr'): 'Helsinki-NLP/opus-mt-sn-fr',
                     ('sn', 'sv'): 'Helsinki-NLP/opus-mt-sn-sv', ('sq', 'en'): 'Helsinki-NLP/opus-mt-sq-en',
                     ('sq', 'es'): 'Helsinki-NLP/opus-mt-sq-es', ('sq', 'sv'): 'Helsinki-NLP/opus-mt-sq-sv',
                     ('srn', 'en'): 'Helsinki-NLP/opus-mt-srn-en', ('srn', 'es'): 'Helsinki-NLP/opus-mt-srn-es',
                     ('srn', 'fr'): 'Helsinki-NLP/opus-mt-srn-fr', ('srn', 'sv'): 'Helsinki-NLP/opus-mt-srn-sv',
                     ('ss', 'en'): 'Helsinki-NLP/opus-mt-ss-en', ('ssp', 'es'): 'Helsinki-NLP/opus-mt-ssp-es',
                     ('st', 'en'): 'Helsinki-NLP/opus-mt-st-en', ('st', 'es'): 'Helsinki-NLP/opus-mt-st-es',
                     ('st', 'fi'): 'Helsinki-NLP/opus-mt-st-fi', ('st', 'fr'): 'Helsinki-NLP/opus-mt-st-fr',
                     ('st', 'sv'): 'Helsinki-NLP/opus-mt-st-sv', ('sv', 'NORWAY'): 'Helsinki-NLP/opus-mt-sv-NORWAY',
                     ('sv', 'ZH'): 'Helsinki-NLP/opus-mt-sv-ZH', ('sv', 'af'): 'Helsinki-NLP/opus-mt-sv-af',
                     ('sv', 'ase'): 'Helsinki-NLP/opus-mt-sv-ase', ('sv', 'bcl'): 'Helsinki-NLP/opus-mt-sv-bcl',
                     ('sv', 'bem'): 'Helsinki-NLP/opus-mt-sv-bem', ('sv', 'bg'): 'Helsinki-NLP/opus-mt-sv-bg',
                     ('sv', 'bi'): 'Helsinki-NLP/opus-mt-sv-bi', ('sv', 'bzs'): 'Helsinki-NLP/opus-mt-sv-bzs',
                     ('sv', 'ceb'): 'Helsinki-NLP/opus-mt-sv-ceb', ('sv', 'chk'): 'Helsinki-NLP/opus-mt-sv-chk',
                     ('sv', 'crs'): 'Helsinki-NLP/opus-mt-sv-crs', ('sv', 'cs'): 'Helsinki-NLP/opus-mt-sv-cs',
                     ('sv', 'ee'): 'Helsinki-NLP/opus-mt-sv-ee', ('sv', 'efi'): 'Helsinki-NLP/opus-mt-sv-efi',
                     ('sv', 'el'): 'Helsinki-NLP/opus-mt-sv-el', ('sv', 'en'): 'Helsinki-NLP/opus-mt-sv-en',
                     ('sv', 'eo'): 'Helsinki-NLP/opus-mt-sv-eo', ('sv', 'es'): 'Helsinki-NLP/opus-mt-sv-es',
                     ('sv', 'et'): 'Helsinki-NLP/opus-mt-sv-et', ('sv', 'fi'): 'Helsinki-NLP/opus-mt-sv-fi',
                     ('sv', 'fj'): 'Helsinki-NLP/opus-mt-sv-fj', ('sv', 'fr'): 'Helsinki-NLP/opus-mt-sv-fr',
                     ('sv', 'gaa'): 'Helsinki-NLP/opus-mt-sv-gaa', ('sv', 'gil'): 'Helsinki-NLP/opus-mt-sv-gil',
                     ('sv', 'guw'): 'Helsinki-NLP/opus-mt-sv-guw', ('sv', 'ha'): 'Helsinki-NLP/opus-mt-sv-ha',
                     ('sv', 'he'): 'Helsinki-NLP/opus-mt-sv-he', ('sv', 'hil'): 'Helsinki-NLP/opus-mt-sv-hil',
                     ('sv', 'ho'): 'Helsinki-NLP/opus-mt-sv-ho', ('sv', 'hr'): 'Helsinki-NLP/opus-mt-sv-hr',
                     ('sv', 'ht'): 'Helsinki-NLP/opus-mt-sv-ht', ('sv', 'hu'): 'Helsinki-NLP/opus-mt-sv-hu',
                     ('sv', 'id'): 'Helsinki-NLP/opus-mt-sv-id', ('sv', 'ig'): 'Helsinki-NLP/opus-mt-sv-ig',
                     ('sv', 'ilo'): 'Helsinki-NLP/opus-mt-sv-ilo', ('sv', 'is'): 'Helsinki-NLP/opus-mt-sv-is',
                     ('sv', 'iso'): 'Helsinki-NLP/opus-mt-sv-iso', ('sv', 'kg'): 'Helsinki-NLP/opus-mt-sv-kg',
                     ('sv', 'kqn'): 'Helsinki-NLP/opus-mt-sv-kqn', ('sv', 'kwy'): 'Helsinki-NLP/opus-mt-sv-kwy',
                     ('sv', 'lg'): 'Helsinki-NLP/opus-mt-sv-lg', ('sv', 'ln'): 'Helsinki-NLP/opus-mt-sv-ln',
                     ('sv', 'lu'): 'Helsinki-NLP/opus-mt-sv-lu', ('sv', 'lua'): 'Helsinki-NLP/opus-mt-sv-lua',
                     ('sv', 'lue'): 'Helsinki-NLP/opus-mt-sv-lue', ('sv', 'lus'): 'Helsinki-NLP/opus-mt-sv-lus',
                     ('sv', 'lv'): 'Helsinki-NLP/opus-mt-sv-lv', ('sv', 'mfe'): 'Helsinki-NLP/opus-mt-sv-mfe',
                     ('sv', 'mh'): 'Helsinki-NLP/opus-mt-sv-mh', ('sv', 'mos'): 'Helsinki-NLP/opus-mt-sv-mos',
                     ('sv', 'mt'): 'Helsinki-NLP/opus-mt-sv-mt', ('sv', 'niu'): 'Helsinki-NLP/opus-mt-sv-niu',
                     ('sv', 'nl'): 'Helsinki-NLP/opus-mt-sv-nl', ('sv', 'no'): 'Helsinki-NLP/opus-mt-sv-no',
                     ('sv', 'nso'): 'Helsinki-NLP/opus-mt-sv-nso', ('sv', 'ny'): 'Helsinki-NLP/opus-mt-sv-ny',
                     ('sv', 'pag'): 'Helsinki-NLP/opus-mt-sv-pag', ('sv', 'pap'): 'Helsinki-NLP/opus-mt-sv-pap',
                     ('sv', 'pis'): 'Helsinki-NLP/opus-mt-sv-pis', ('sv', 'pon'): 'Helsinki-NLP/opus-mt-sv-pon',
                     ('sv', 'rnd'): 'Helsinki-NLP/opus-mt-sv-rnd', ('sv', 'ro'): 'Helsinki-NLP/opus-mt-sv-ro',
                     ('sv', 'ru'): 'Helsinki-NLP/opus-mt-sv-ru', ('sv', 'run'): 'Helsinki-NLP/opus-mt-sv-run',
                     ('sv', 'rw'): 'Helsinki-NLP/opus-mt-sv-rw', ('sv', 'sg'): 'Helsinki-NLP/opus-mt-sv-sg',
                     ('sv', 'sk'): 'Helsinki-NLP/opus-mt-sv-sk', ('sv', 'sl'): 'Helsinki-NLP/opus-mt-sv-sl',
                     ('sv', 'sm'): 'Helsinki-NLP/opus-mt-sv-sm', ('sv', 'sn'): 'Helsinki-NLP/opus-mt-sv-sn',
                     ('sv', 'sq'): 'Helsinki-NLP/opus-mt-sv-sq', ('sv', 'srn'): 'Helsinki-NLP/opus-mt-sv-srn',
                     ('sv', 'st'): 'Helsinki-NLP/opus-mt-sv-st', ('sv', 'sv'): 'Helsinki-NLP/opus-mt-sv-sv',
                     ('sv', 'swc'): 'Helsinki-NLP/opus-mt-sv-swc', ('sv', 'th'): 'Helsinki-NLP/opus-mt-sv-th',
                     ('sv', 'tiv'): 'Helsinki-NLP/opus-mt-sv-tiv', ('sv', 'tll'): 'Helsinki-NLP/opus-mt-sv-tll',
                     ('sv', 'tn'): 'Helsinki-NLP/opus-mt-sv-tn', ('sv', 'to'): 'Helsinki-NLP/opus-mt-sv-to',
                     ('sv', 'toi'): 'Helsinki-NLP/opus-mt-sv-toi', ('sv', 'tpi'): 'Helsinki-NLP/opus-mt-sv-tpi',
                     ('sv', 'ts'): 'Helsinki-NLP/opus-mt-sv-ts', ('sv', 'tum'): 'Helsinki-NLP/opus-mt-sv-tum',
                     ('sv', 'tvl'): 'Helsinki-NLP/opus-mt-sv-tvl', ('sv', 'tw'): 'Helsinki-NLP/opus-mt-sv-tw',
                     ('sv', 'ty'): 'Helsinki-NLP/opus-mt-sv-ty', ('sv', 'uk'): 'Helsinki-NLP/opus-mt-sv-uk',
                     ('sv', 'umb'): 'Helsinki-NLP/opus-mt-sv-umb', ('sv', 've'): 'Helsinki-NLP/opus-mt-sv-ve',
                     ('sv', 'war'): 'Helsinki-NLP/opus-mt-sv-war', ('sv', 'wls'): 'Helsinki-NLP/opus-mt-sv-wls',
                     ('sv', 'xh'): 'Helsinki-NLP/opus-mt-sv-xh', ('sv', 'yap'): 'Helsinki-NLP/opus-mt-sv-yap',
                     ('sv', 'yo'): 'Helsinki-NLP/opus-mt-sv-yo', ('sv', 'zne'): 'Helsinki-NLP/opus-mt-sv-zne',
                     ('swc', 'en'): 'Helsinki-NLP/opus-mt-swc-en', ('swc', 'es'): 'Helsinki-NLP/opus-mt-swc-es',
                     ('swc', 'fi'): 'Helsinki-NLP/opus-mt-swc-fi', ('swc', 'fr'): 'Helsinki-NLP/opus-mt-swc-fr',
                     ('swc', 'sv'): 'Helsinki-NLP/opus-mt-swc-sv', ('taw', 'en'): 'Helsinki-NLP/opus-mt-taw-en',
                     ('th', 'en'): 'Helsinki-NLP/opus-mt-th-en', ('th', 'fr'): 'Helsinki-NLP/opus-mt-th-fr',
                     ('ti', 'en'): 'Helsinki-NLP/opus-mt-ti-en', ('tiv', 'en'): 'Helsinki-NLP/opus-mt-tiv-en',
                     ('tiv', 'fr'): 'Helsinki-NLP/opus-mt-tiv-fr', ('tiv', 'sv'): 'Helsinki-NLP/opus-mt-tiv-sv',
                     ('tl', 'de'): 'Helsinki-NLP/opus-mt-tl-de', ('tl', 'en'): 'Helsinki-NLP/opus-mt-tl-en',
                     ('tl', 'es'): 'Helsinki-NLP/opus-mt-tl-es', ('tl', 'pt'): 'Helsinki-NLP/opus-mt-tl-pt',
                     ('tll', 'en'): 'Helsinki-NLP/opus-mt-tll-en', ('tll', 'es'): 'Helsinki-NLP/opus-mt-tll-es',
                     ('tll', 'fi'): 'Helsinki-NLP/opus-mt-tll-fi', ('tll', 'fr'): 'Helsinki-NLP/opus-mt-tll-fr',
                     ('tll', 'sv'): 'Helsinki-NLP/opus-mt-tll-sv', ('tn', 'en'): 'Helsinki-NLP/opus-mt-tn-en',
                     ('tn', 'es'): 'Helsinki-NLP/opus-mt-tn-es', ('tn', 'fr'): 'Helsinki-NLP/opus-mt-tn-fr',
                     ('tn', 'sv'): 'Helsinki-NLP/opus-mt-tn-sv', ('to', 'en'): 'Helsinki-NLP/opus-mt-to-en',
                     ('to', 'es'): 'Helsinki-NLP/opus-mt-to-es', ('to', 'fr'): 'Helsinki-NLP/opus-mt-to-fr',
                     ('to', 'sv'): 'Helsinki-NLP/opus-mt-to-sv', ('toi', 'en'): 'Helsinki-NLP/opus-mt-toi-en',
                     ('toi', 'es'): 'Helsinki-NLP/opus-mt-toi-es', ('toi', 'fi'): 'Helsinki-NLP/opus-mt-toi-fi',
                     ('toi', 'fr'): 'Helsinki-NLP/opus-mt-toi-fr', ('toi', 'sv'): 'Helsinki-NLP/opus-mt-toi-sv',
                     ('tpi', 'en'): 'Helsinki-NLP/opus-mt-tpi-en', ('tpi', 'sv'): 'Helsinki-NLP/opus-mt-tpi-sv',
                     ('tr', 'ar'): 'Helsinki-NLP/opus-mt-tr-ar', ('tr', 'az'): 'Helsinki-NLP/opus-mt-tr-az',
                     ('tr', 'en'): 'Helsinki-NLP/opus-mt-tr-en', ('tr', 'eo'): 'Helsinki-NLP/opus-mt-tr-eo',
                     ('tr', 'es'): 'Helsinki-NLP/opus-mt-tr-es', ('tr', 'fr'): 'Helsinki-NLP/opus-mt-tr-fr',
                     ('tr', 'lt'): 'Helsinki-NLP/opus-mt-tr-lt', ('tr', 'sv'): 'Helsinki-NLP/opus-mt-tr-sv',
                     ('tr', 'uk'): 'Helsinki-NLP/opus-mt-tr-uk', ('trk', 'en'): 'Helsinki-NLP/opus-mt-trk-en',
                     ('ts', 'en'): 'Helsinki-NLP/opus-mt-ts-en', ('ts', 'es'): 'Helsinki-NLP/opus-mt-ts-es',
                     ('ts', 'fi'): 'Helsinki-NLP/opus-mt-ts-fi', ('ts', 'fr'): 'Helsinki-NLP/opus-mt-ts-fr',
                     ('ts', 'sv'): 'Helsinki-NLP/opus-mt-ts-sv', ('tum', 'en'): 'Helsinki-NLP/opus-mt-tum-en',
                     ('tum', 'es'): 'Helsinki-NLP/opus-mt-tum-es', ('tum', 'fr'): 'Helsinki-NLP/opus-mt-tum-fr',
                     ('tum', 'sv'): 'Helsinki-NLP/opus-mt-tum-sv', ('tvl', 'en'): 'Helsinki-NLP/opus-mt-tvl-en',
                     ('tvl', 'es'): 'Helsinki-NLP/opus-mt-tvl-es', ('tvl', 'fi'): 'Helsinki-NLP/opus-mt-tvl-fi',
                     ('tvl', 'fr'): 'Helsinki-NLP/opus-mt-tvl-fr', ('tvl', 'sv'): 'Helsinki-NLP/opus-mt-tvl-sv',
                     ('tw', 'es'): 'Helsinki-NLP/opus-mt-tw-es', ('tw', 'fi'): 'Helsinki-NLP/opus-mt-tw-fi',
                     ('tw', 'fr'): 'Helsinki-NLP/opus-mt-tw-fr', ('tw', 'sv'): 'Helsinki-NLP/opus-mt-tw-sv',
                     ('ty', 'es'): 'Helsinki-NLP/opus-mt-ty-es', ('ty', 'fi'): 'Helsinki-NLP/opus-mt-ty-fi',
                     ('ty', 'fr'): 'Helsinki-NLP/opus-mt-ty-fr', ('ty', 'sv'): 'Helsinki-NLP/opus-mt-ty-sv',
                     ('tzo', 'es'): 'Helsinki-NLP/opus-mt-tzo-es', ('uk', 'bg'): 'Helsinki-NLP/opus-mt-uk-bg',
                     ('uk', 'ca'): 'Helsinki-NLP/opus-mt-uk-ca', ('uk', 'cs'): 'Helsinki-NLP/opus-mt-uk-cs',
                     ('uk', 'de'): 'Helsinki-NLP/opus-mt-uk-de', ('uk', 'en'): 'Helsinki-NLP/opus-mt-uk-en',
                     ('uk', 'es'): 'Helsinki-NLP/opus-mt-uk-es', ('uk', 'fi'): 'Helsinki-NLP/opus-mt-uk-fi',
                     ('uk', 'fr'): 'Helsinki-NLP/opus-mt-uk-fr', ('uk', 'he'): 'Helsinki-NLP/opus-mt-uk-he',
                     ('uk', 'hu'): 'Helsinki-NLP/opus-mt-uk-hu', ('uk', 'it'): 'Helsinki-NLP/opus-mt-uk-it',
                     ('uk', 'nl'): 'Helsinki-NLP/opus-mt-uk-nl', ('uk', 'no'): 'Helsinki-NLP/opus-mt-uk-no',
                     ('uk', 'pl'): 'Helsinki-NLP/opus-mt-uk-pl', ('uk', 'pt'): 'Helsinki-NLP/opus-mt-uk-pt',
                     ('uk', 'ru'): 'Helsinki-NLP/opus-mt-uk-ru', ('uk', 'sh'): 'Helsinki-NLP/opus-mt-uk-sh',
                     ('uk', 'sl'): 'Helsinki-NLP/opus-mt-uk-sl', ('uk', 'sv'): 'Helsinki-NLP/opus-mt-uk-sv',
                     ('uk', 'tr'): 'Helsinki-NLP/opus-mt-uk-tr', ('umb', 'en'): 'Helsinki-NLP/opus-mt-umb-en',
                     ('ur', 'en'): 'Helsinki-NLP/opus-mt-ur-en', ('urj', 'en'): 'Helsinki-NLP/opus-mt-urj-en',
                     ('urj', 'urj'): 'Helsinki-NLP/opus-mt-urj-urj', ('ve', 'en'): 'Helsinki-NLP/opus-mt-ve-en',
                     ('ve', 'es'): 'Helsinki-NLP/opus-mt-ve-es', ('vi', 'de'): 'Helsinki-NLP/opus-mt-vi-de',
                     ('vi', 'en'): 'Helsinki-NLP/opus-mt-vi-en', ('vi', 'eo'): 'Helsinki-NLP/opus-mt-vi-eo',
                     ('vi', 'es'): 'Helsinki-NLP/opus-mt-vi-es', ('vi', 'fr'): 'Helsinki-NLP/opus-mt-vi-fr',
                     ('vi', 'it'): 'Helsinki-NLP/opus-mt-vi-it', ('vi', 'ru'): 'Helsinki-NLP/opus-mt-vi-ru',
                     ('vsl', 'es'): 'Helsinki-NLP/opus-mt-vsl-es', ('wa', 'en'): 'Helsinki-NLP/opus-mt-wa-en',
                     ('wal', 'en'): 'Helsinki-NLP/opus-mt-wal-en', ('war', 'en'): 'Helsinki-NLP/opus-mt-war-en',
                     ('war', 'es'): 'Helsinki-NLP/opus-mt-war-es', ('war', 'fi'): 'Helsinki-NLP/opus-mt-war-fi',
                     ('war', 'fr'): 'Helsinki-NLP/opus-mt-war-fr', ('war', 'sv'): 'Helsinki-NLP/opus-mt-war-sv',
                     ('wls', 'en'): 'Helsinki-NLP/opus-mt-wls-en', ('wls', 'fr'): 'Helsinki-NLP/opus-mt-wls-fr',
                     ('wls', 'sv'): 'Helsinki-NLP/opus-mt-wls-sv', ('xh', 'en'): 'Helsinki-NLP/opus-mt-xh-en',
                     ('xh', 'es'): 'Helsinki-NLP/opus-mt-xh-es', ('xh', 'fr'): 'Helsinki-NLP/opus-mt-xh-fr',
                     ('xh', 'sv'): 'Helsinki-NLP/opus-mt-xh-sv', ('yap', 'en'): 'Helsinki-NLP/opus-mt-yap-en',
                     ('yap', 'fr'): 'Helsinki-NLP/opus-mt-yap-fr', ('yap', 'sv'): 'Helsinki-NLP/opus-mt-yap-sv',
                     ('yo', 'en'): 'Helsinki-NLP/opus-mt-yo-en', ('yo', 'es'): 'Helsinki-NLP/opus-mt-yo-es',
                     ('yo', 'fi'): 'Helsinki-NLP/opus-mt-yo-fi', ('yo', 'fr'): 'Helsinki-NLP/opus-mt-yo-fr',
                     ('yo', 'sv'): 'Helsinki-NLP/opus-mt-yo-sv', ('zai', 'es'): 'Helsinki-NLP/opus-mt-zai-es',
                     ('zh', 'bg'): 'Helsinki-NLP/opus-mt-zh-bg', ('zh', 'de'): 'Helsinki-NLP/opus-mt-zh-de',
                     ('zh', 'en'): 'Helsinki-NLP/opus-mt-zh-en', ('zh', 'fi'): 'Helsinki-NLP/opus-mt-zh-fi',
                     ('zh', 'he'): 'Helsinki-NLP/opus-mt-zh-he', ('zh', 'it'): 'Helsinki-NLP/opus-mt-zh-it',
                     ('zh', 'ms'): 'Helsinki-NLP/opus-mt-zh-ms', ('zh', 'nl'): 'Helsinki-NLP/opus-mt-zh-nl',
                     ('zh', 'sv'): 'Helsinki-NLP/opus-mt-zh-sv', ('zh', 'uk'): 'Helsinki-NLP/opus-mt-zh-uk',
                     ('zh', 'vi'): 'Helsinki-NLP/opus-mt-zh-vi', ('zle', 'en'): 'Helsinki-NLP/opus-mt-zle-en',
                     ('zle', 'zle'): 'Helsinki-NLP/opus-mt-zle-zle', ('zls', 'en'): 'Helsinki-NLP/opus-mt-zls-en',
                     ('zls', 'zls'): 'Helsinki-NLP/opus-mt-zls-zls', ('zlw', 'en'): 'Helsinki-NLP/opus-mt-zlw-en',
                     ('zlw', 'fiu'): 'Helsinki-NLP/opus-mt-zlw-fiu', ('zlw', 'zlw'): 'Helsinki-NLP/opus-mt-zlw-zlw',
                     ('zne', 'es'): 'Helsinki-NLP/opus-mt-zne-es', ('zne', 'fi'): 'Helsinki-NLP/opus-mt-zne-fi',
                     ('zne', 'fr'): 'Helsinki-NLP/opus-mt-zne-fr', ('zne', 'sv'): 'Helsinki-NLP/opus-mt-zne-sv'}

# from https://github.com/joke2k/faker/blob/master/faker/providers/person/en/__init__.py which is under the MIT License
# not an exausthive list. just to do some filtering for civil_comment.
first_names = {
    'Aaliyah', 'Abagail', 'Abbey', 'Abbie', 'Abbigail', 'Abby', 'Abigail',
    'Abigale', 'Abigayle', 'Abril', 'Achsah', 'Ada', 'Adah', 'Adaline',
    'Adalyn', 'Adalynn', 'Adamaris', 'Adda', 'Addie', 'Addison', 'Addisyn',
    'Addyson', 'Adel', 'Adela', 'Adelaide', 'Adele', 'Adelia', 'Adelina',
    'Adeline', 'Adell', 'Adella', 'Adelle', 'Adelyn', 'Adelynn', 'Adilene',
    'Adina', 'Adison', 'Adline', 'Adria', 'Adriana', 'Adriane', 'Adrianna',
    'Adrianne', 'Adriene', 'Adrienne', 'Adyson', 'Affie', 'Afton', 'Agatha',
    'Aggie', 'Agnes', 'Agness', 'Agusta', 'Aida', 'Aileen', 'Ailene',
    'Aili', 'Aimee', 'Ainsley', 'Aisha', 'Aiyana', 'Aiyanna', 'Aja',
    'Akeelah', 'Akira', 'Ala', 'Alabama', 'Alaina', 'Alana', 'Alani',
    'Alanna', 'Alannah', 'Alaya', 'Alayna', 'Alba', 'Alberta', 'Albertha',
    'Albertina', 'Albertine', 'Albina', 'Alcie', 'Alda', 'Aldona', 'Aleah',
    'Alease', 'Alecia', 'Aleen', 'Aleena', 'Alejandra', 'Alena', 'Alene',
    'Alesha', 'Alesia', 'Alessandra', 'Aleta', 'Aletha', 'Alethea', 'Alex',
    'Alexa', 'Alexandr', 'Alexandra', 'Alexandrea', 'Alexandria', 'Alexia',
    'Alexina', 'Alexis', 'Alexus', 'Alexys', 'Alfreda', 'Alia', 'Aliana',
    'Alice', 'Alicia', 'Alida', 'Alina', 'Aline', 'Alisa', 'Alisha',
    'Alison', 'Alissa', 'Alisson', 'Alivia', 'Aliya', 'Aliyah', 'Aliza',
    'Alize', 'Alla', 'Allean', 'Alleen', 'Allena', 'Allene', 'Allie',
    'Alline', 'Allison', 'Allisson', 'Ally', 'Allyson', 'Allyssa', 'Alma',
    'Almeda', 'Almedia', 'Almeta', 'Almina', 'Almira', 'Almyra', 'Aloma',
    'Alondra', 'Alpha', 'Alphonsine', 'Alta', 'Altha', 'Althea', 'Altie',
    'Alvena', 'Alvera', 'Alverda', 'Alverta', 'Alvina', 'Alvira', 'Alwilda',
    'Alwina', 'Alwine', 'Alyce', 'Alycia', 'Alys', 'Alysa', 'Alyse',
    'Alysha', 'Alysia', 'Alyson', 'Alyssa', 'Alyssia', 'Alyvia', 'Alzina',
    'Ama', 'Amalia', 'Amalie', 'Amanda', 'Amani', 'Amara', 'Amari',
    'Amaris', 'Amaya', 'Amber', 'Amberly', 'Amelia', 'Amelie', 'America',
    'Amey', 'Ami', 'Amiah', 'Amie', 'Amina', 'Amira', 'Amirah', 'Amiya',
    'Amiyah', 'Amma', 'Ammie', 'Amparo', 'Amy', 'Amya', 'Ana', 'Anabel',
    'Anabella', 'Anabelle', 'Anahi', 'Anais', 'Analia', 'Anastacia',
    'Anastasia', 'Anaya', 'Andra', 'Andrea', 'Andria', 'Angel', 'Angela',
    'Angele', 'Angeles', 'Angelia', 'Angelic', 'Angelica', 'Angelina',
    'Angeline', 'Angelique', 'Angelita', 'Angella', 'Angie', 'Anice',
    'Anie', 'Anika', 'Anissa', 'Anita', 'Anitra', 'Aniya', 'Aniyah',
    'Anjali', 'Anjanette', 'Anjelica', 'Ann', 'Anna', 'Annabel', 'Annabell',
    'Annabella', 'Annabelle', 'Annalise', 'Annamae', 'Annamarie', 'Anne',
    'Anneliese', 'Annemarie', 'Anner', 'Annetta', 'Annette', 'Annice',
    'Annie', 'Annika', 'Annis', 'Annmarie', 'Anona', 'Ansley', 'Antionette',
    'Antoinette', 'Antonetta', 'Antonette', 'Antonia', 'Antonina', 'Anya',
    'April', 'Ara', 'Arabella', 'Araceli', 'Aracely', 'Arah', 'Araminta',
    'Ardath', 'Ardelia', 'Ardell', 'Ardella', 'Ardelle', 'Arden', 'Ardeth',
    'Ardis', 'Ardith', 'Ardyce', 'Areli', 'Arely', 'Aretha', 'Argie',
    'Aria', 'Ariana', 'Ariane', 'Arianna', 'Arie', 'Ariel', 'Ariella',
    'Arielle', 'Arietta', 'Arizona', 'Arkie', 'Arla', 'Arleen', 'Arlena',
    'Arlene', 'Arleth', 'Arletta', 'Arley', 'Arlie', 'Arline', 'Arly',
    'Arlyne', 'Armani', 'Armida', 'Arminda', 'Arminta', 'Arnetta', 'Arra',
    'Arrie', 'Arta', 'Artelia', 'Arvilla', 'Aryana', 'Aryanna', 'Asha',
    'Ashanti', 'Ashely', 'Ashlea', 'Ashlee', 'Ashleigh', 'Ashley', 'Ashli',
    'Ashlie', 'Ashly', 'Ashlyn', 'Ashlynn', 'Ashtyn', 'Asia', 'Ason',
    'Aspen', 'Assunta', 'Astrid', 'Atha', 'Athena', 'Attie', 'Aubree',
    'Aubrey', 'Aubrie', 'Audie', 'Audra', 'Audrey', 'Audriana', 'Audrianna',
    'Audrina', 'Audry', 'Augusta', 'Augustina', 'Aura', 'Aurelia',
    'Aurilla', 'Aurora', 'Aurore', 'Autumn', 'Ava', 'Avah', 'Averi',
    'Averie', 'Avie', 'Avis', 'Ayana', 'Ayanna', 'Ayesha', 'Ayla', 'Ayleen',
    'Aylin', 'Azalee', 'Azaria', 'Azariah', 'Azul', 'Azzie', 'Babette',
    'Baby', 'Bailee', 'Bailey', 'Bama', 'Bambi', 'Barb', 'Barbara',
    'Barbie', 'Barbra', 'Baylee', 'Baylie', 'Bea', 'Beadie', 'Beatrice',
    'Beatrix', 'Beatriz', 'Beaulah', 'Bebe', 'Beckie', 'Becky', 'Beda',
    'Bee', 'Belen', 'Belia', 'Belinda', 'Bell', 'Bella', 'Belle', 'Belva',
    'Bena', 'Benita', 'Bennie', 'Berdie', 'Berenice', 'Bernadette',
    'Bernadine', 'Bernardine', 'Berneice', 'Bernetta', 'Bernice',
    'Berniece', 'Bernita', 'Berta', 'Bertha', 'Bertie', 'Bertina', 'Beryl',
    'Bess', 'Besse', 'Bessie', 'Beth', 'Betha', 'Bethann', 'Bethany',
    'Bethel', 'Bethzy', 'Betsey', 'Betsy', 'Bette', 'Bettie', 'Bettina',
    'Betty', 'Bettye', 'Bettyjane', 'Bettylou', 'Beula', 'Beulah', 'Bev',
    'Beverlee', 'Beverley', 'Beverly', 'Beyonce', 'Bianca', 'Biddie',
    'Billie', 'Billy', 'Billye', 'Bina', 'Bird', 'Birdella', 'Birdie',
    'Birtha', 'Birtie', 'Blair', 'Blake', 'Blanca', 'Blanch', 'Blanche',
    'Blanchie', 'Blossom', 'Bobbi', 'Bobbie', 'Bobby', 'Bobbye', 'Bonita',
    'Bonnie', 'Bonny', 'Braelyn', 'Brande', 'Brandee', 'Brandi', 'Brandie',
    'Brandon', 'Brandy', 'Brea', 'Breana', 'Breann', 'Breanna', 'Breanne',
    'Bree', 'Brenda', 'Brenna', 'Breonna', 'Brett', 'Bria', 'Briana',
    'Brianda', 'Brianna', 'Brianne', 'Bridget', 'Bridgett', 'Bridgette',
    'Brielle', 'Brigette', 'Brigid', 'Brigitte', 'Briley', 'Brinda',
    'Brinley', 'Brionna', 'Brisa', 'Bristol', 'Britany', 'Britney',
    'Britni', 'Britny', 'Britt', 'Britta', 'Brittaney', 'Brittani',
    'Brittanie', 'Brittany', 'Brittnay', 'Brittnee', 'Brittney', 'Brittni',
    'Brittnie', 'Brittny', 'Brook', 'Brooke', 'Brooklyn', 'Brooklynn',
    'Bryana', 'Bryanna', 'Brylee', 'Bryn', 'Brynlee', 'Brynn', 'Buelah',
    'Buena', 'Buffy', 'Bula', 'Bulah', 'Buna', 'Burnice', 'Byrd', 'Byrdie',
    'Caddie', 'Cadence', 'Cailyn', 'Caitlin', 'Caitlyn', 'Caitlynn',
    'Caldonia', 'Caleigh', 'Cali', 'Calista', 'Calla', 'Calleigh', 'Callie',
    'Cambria', 'Cameron', 'Cami', 'Camila', 'Camilla', 'Camille', 'Camisha',
    'Cammie', 'Campbell', 'Camryn', 'Candace', 'Candi', 'Candice',
    'Candida', 'Candis', 'Candy', 'Candyce', 'Cannie', 'Capitola', 'Cappie',
    'Caprice', 'Cara', 'Caren', 'Carey', 'Cari', 'Carie', 'Carin', 'Carina',
    'Carisa', 'Carissa', 'Carla', 'Carlee', 'Carleen', 'Carleigh',
    'Carlene', 'Carley', 'Carli', 'Carlie', 'Carlota', 'Carlotta', 'Carly',
    'Carlyn', 'Carma', 'Carmel', 'Carmela', 'Carmelita', 'Carmella',
    'Carmen', 'Caro', 'Carol', 'Carolann', 'Carole', 'Carolee', 'Carolina',
    'Caroline', 'Carolyn', 'Carolyne', 'Carolynn', 'Caron', 'Carra',
    'Carri', 'Carrie', 'Carrol', 'Carroll', 'Carry', 'Carson', 'Cary',
    'Caryl', 'Caryn', 'Casandra', 'Casey', 'Casie', 'Cassandra', 'Cassidy',
    'Cassie', 'Cassondra', 'Catalina', 'Catharine', 'Catherine', 'Cathern',
    'Cathey', 'Cathi', 'Cathie', 'Cathleen', 'Cathrine', 'Cathryn', 'Cathy',
    'Catina', 'Catrina', 'Caydence', 'Cayla', 'Caylee', 'Cecelia', 'Cecile',
    'Cecilia', 'Cecily', 'Ceil', 'Celena', 'Celesta', 'Celeste', 'Celestia',
    'Celestine', 'Celia', 'Celie', 'Celina', 'Celine', 'Cena', 'Ceola',
    'Chaka', 'Chana', 'Chanda', 'Chandler', 'Chandra', 'Chanel', 'Chanelle',
    'Chaney', 'Chanie', 'Channie', 'Channing', 'Chantal', 'Chante',
    'Chantel', 'Chantelle', 'Charissa', 'Charisse', 'Charity', 'Charla',
    'Charlee', 'Charleen', 'Charlene', 'Charley', 'Charlie', 'Charline',
    'Charlize', 'Charlotta', 'Charlotte', 'Charlottie', 'Charlsie',
    'Charmaine', 'Charolette', 'Chase', 'Chasity', 'Chastity', 'Chaya',
    'Chelsea', 'Chelsey', 'Chelsi', 'Chelsie', 'Chelsy', 'Cher', 'Cherelle',
    'Cheri', 'Cherie', 'Cherilyn', 'Cherise', 'Cherish', 'Cherrelle',
    'Cherri', 'Cherrie', 'Cherry', 'Cherryl', 'Cheryl', 'Cheryle',
    'Cheryll', 'Chessie', 'Chestina', 'Cheyanne', 'Cheyenne', 'Chimere',
    'China', 'Chiquita', 'Chloe', 'Chloie', 'Chris', 'Chrissie', 'Chrissy',
    'Christa', 'Christal', 'Christeen', 'Christel', 'Christen', 'Christena',
    'Christene', 'Christi', 'Christian', 'Christiana', 'Christie',
    'Christin', 'Christina', 'Christine', 'Christy', 'Chrystal', 'Chyna',
    'Chynna', 'Ciara', 'Ciarra', 'Cicely', 'Cielo', 'Ciera', 'Cierra',
    'Ciji', 'Cilla', 'Cinda', 'Cindi', 'Cindy', 'Cinnamon', 'Cinthia',
    'Citlali', 'Citlalli', 'Clair', 'Claire', 'Clara', 'Clarabelle',
    'Clare', 'Claribel', 'Clarice', 'Clarinda', 'Clarine', 'Clarisa',
    'Clarissa', 'Classie', 'Claudette', 'Claudia', 'Claudie', 'Claudine',
    'Cleda', 'Clella', 'Clem', 'Clemence', 'Clementina', 'Clementine',
    'Clemie', 'Clemma', 'Clemmie', 'Cleo', 'Cleola', 'Cleone', 'Cleora',
    'Cleta', 'Cleva', 'Clevie', 'Cliffie', 'Cloe', 'Clora', 'Clotilda',
    'Clotilde', 'Clyda', 'Clydie', 'Clytie', 'Coleen', 'Coletta', 'Colette',
    'Colleen', 'Collette', 'Columbia', 'Concepcion', 'Concetta', 'Concha',
    'Connie', 'Constance', 'Consuela', 'Consuelo', 'Contina', 'Cora',
    'Coraima', 'Coral', 'Coralie', 'Corda', 'Cordelia', 'Cordella',
    'Cordia', 'Cordie', 'Corean', 'Corene', 'Coretta', 'Corey', 'Cori',
    'Corie', 'Corina', 'Corine', 'Corinna', 'Corinne', 'Corliss',
    'Cornelia', 'Cornie', 'Corrie', 'Corrina', 'Corrine', 'Cortney', 'Cory',
    'Courtney', 'Creola', 'Cressie', 'Crete', 'Crissie', 'Crissy', 'Crista',
    'Cristal', 'Cristen', 'Cristi', 'Cristin', 'Cristina', 'Cristine',
    'Cristy', 'Cruz', 'Crysta', 'Crystal', 'Cuba', 'Cydney', 'Cyndi',
    'Cyntha', 'Cynthia', 'Dafne', 'Dagmar', 'Dagny', 'Dahlia', 'Daija',
    'Daijah', 'Daisey', 'Daisha', 'Daisie', 'Daisy', 'Daisye', 'Daja',
    'Dakota', 'Dale', 'Dalia', 'Dallas', 'Damaris', 'Dana', 'Danae',
    'Daneen', 'Danelle', 'Danette', 'Dani', 'Dania', 'Danica', 'Daniela',
    'Daniele', 'Daniella', 'Danielle', 'Danika', 'Danita', 'Danna',
    'Dannie', 'Dannielle', 'Danyel', 'Danyell', 'Danyelle', 'Daphne',
    'Dara', 'Darby', 'Darci', 'Darcie', 'Darcy', 'Daria', 'Darian',
    'Dariana', 'Darla', 'Darleen', 'Darlene', 'Darline', 'Darlyne', 'Dasia',
    'Davina', 'Dawn', 'Dawna', 'Dawne', 'Dayami', 'Dayana', 'Dayanara',
    'Dayle', 'Dayna', 'Dayse', 'Deana', 'Deandra', 'Deann', 'Deanna',
    'Deanne', 'Deasia', 'Deb', 'Debbi', 'Debbie', 'Debbra', 'Debby',
    'Debera', 'Debi', 'Debora', 'Deborah', 'Deborrah', 'Debra', 'Debrah',
    'Debroah', 'Dedra', 'Dee', 'Deeann', 'Deedee', 'Deena', 'Deetta',
    'Deidra', 'Deidre', 'Deirdre', 'Deja', 'Dejah', 'Delaney', 'Delcie',
    'Delfina', 'Delia', 'Deliah', 'Delila', 'Delilah', 'Delina', 'Delinda',
    'Delisa', 'Dell', 'Della', 'Dellar', 'Delle', 'Dellia', 'Dellie',
    'Delma', 'Delois', 'Delora', 'Delores', 'Deloris', 'Delpha', 'Delphia',
    'Delphine', 'Delsie', 'Delta', 'Dema', 'Demetra', 'Demetria', 'Demi',
    'Dena', 'Deneen', 'Denese', 'Denice', 'Denine', 'Denise', 'Denisha',
    'Denisse', 'Denita', 'Dennie', 'Desirae', 'Desiree', 'Dessa', 'Dessie',
    'Destany', 'Destinee', 'Destiney', 'Destini', 'Destiny', 'Devan',
    'Devin', 'Devon', 'Devyn', 'Dewey', 'Deyanira', 'Dezzie', 'Diamond',
    'Dian', 'Diana', 'Diandra', 'Diane', 'Diann', 'Dianna', 'Dianne',
    'Dicie', 'Dicy', 'Dillie', 'Dimple', 'Dina', 'Dinah', 'Dione', 'Dionne',
    'Dixie', 'Diya', 'Djuana', 'Djuna', 'Docia', 'Dola', 'Dollie', 'Dolly',
    'Dollye', 'Dolores', 'Doloris', 'Domenica', 'Dominga', 'Dominique',
    'Dominque', 'Domonique', 'Dona', 'Donia', 'Donie', 'Donita', 'Donna',
    'Donnie', 'Dora', 'Dorathea', 'Dorathy', 'Dorcas', 'Doreen', 'Dorene',
    'Doretha', 'Doretta', 'Dori', 'Dorinda', 'Dorine', 'Doris', 'Dorla',
    'Dorotha', 'Dorothea', 'Dorothy', 'Dorris', 'Dortha', 'Dorthea',
    'Dorthey', 'Dorthy', 'Dosha', 'Doshia', 'Doshie', 'Dosia', 'Dossie',
    'Dot', 'Dottie', 'Dotty', 'Dove', 'Dovie', 'Drema', 'Drew', 'Drucilla',
    'Drusilla', 'Dulce', 'Dulcie', 'Dusty', 'Dwan', 'Dyan', 'Dylan',
    'Earlean', 'Earlene', 'Earlie', 'Earline', 'Earnestine', 'Eartha',
    'Easter', 'Eathel', 'Ebba', 'Eboni', 'Ebony', 'Echo', 'Eda', 'Eddie',
    'Eden', 'Edie', 'Edith', 'Edla', 'Edmonia', 'Edna', 'Ednah', 'Edra',
    'Edrie', 'Edris', 'Edwina', 'Edyth', 'Edythe', 'Effa', 'Effie',
    'Eileen', 'Eithel', 'Ela', 'Elaina', 'Elaine', 'Elana', 'Elayne',
    'Elba', 'Elberta', 'Elda', 'Eldora', 'Eleanor', 'Eleanora', 'Eleanore',
    'Elease', 'Electa', 'Elena', 'Elenor', 'Elenora', 'Elenore', 'Eleonora',
    'Eleonore', 'Elfie', 'Elfreda', 'Elfrieda', 'Elgie', 'Elia', 'Eliana',
    'Elianna', 'Elida', 'Elinor', 'Elinore', 'Elisa', 'Elisabeth', 'Elise',
    'Elisha', 'Elissa', 'Eliza', 'Elizabet', 'Elizabeth', 'Elizbeth',
    'Elizebeth', 'Ella', 'Ellamae', 'Ellar', 'Elle', 'Ellen', 'Eller',
    'Elliana', 'Ellie', 'Ellyn', 'Elma', 'Elmina', 'Elmira', 'Elmire',
    'Elmyra', 'Elna', 'Elnora', 'Elodie', 'Elois', 'Eloisa', 'Eloise',
    'Elouise', 'Elsa', 'Else', 'Elsie', 'Elta', 'Elva', 'Elvera', 'Elvia',
    'Elvie', 'Elvina', 'Elvira', 'Elwanda', 'Elyse', 'Elyssa', 'Elza',
    'Elzada', 'Ema', 'Emaline', 'Ember', 'Emelia', 'Emelie', 'Emeline',
    'Emely', 'Emerald', 'Emerson', 'Emery', 'Emilee', 'Emilia', 'Emilie',
    'Emily', 'Emma', 'Emmalee', 'Emmaline', 'Emmer', 'Emmie', 'Emmy',
    'Emogene', 'Ena', 'Enid', 'Enola', 'Enriqueta', 'Eola', 'Eppie',
    'Epsie', 'Era', 'Erica', 'Ericka', 'Erie', 'Erika', 'Erin', 'Eris',
    'Erla', 'Erlene', 'Erlinda', 'Erline', 'Erma', 'Ermina', 'Ermine',
    'Erna', 'Ernestina', 'Ernestine', 'Erykah', 'Eryn', 'Esmeralda',
    'Esperanza', 'Essa', 'Essence', 'Essie', 'Esta', 'Estefani',
    'Estefania', 'Estefany', 'Estela', 'Estell', 'Estella', 'Estelle',
    'Ester', 'Esther', 'Estie', 'Estrella', 'Etha', 'Ethel', 'Ethelene',
    'Ethelyn', 'Ether', 'Ethie', 'Ethyl', 'Ethyle', 'Etna', 'Etta', 'Etter',
    'Ettie', 'Eudora', 'Eugenia', 'Eugenie', 'Eula', 'Eulah', 'Eulalia',
    'Eulalie', 'Euna', 'Eunice', 'Euphemia', 'Eura', 'Eva', 'Evalena',
    'Evaline', 'Evalyn', 'Evangelina', 'Evangeline', 'Eve', 'Evelena',
    'Evelin', 'Evelina', 'Eveline', 'Evelyn', 'Evelyne', 'Evelynn', 'Ever',
    'Evette', 'Evia', 'Evie', 'Evita', 'Evon', 'Evonne', 'Exa', 'Exie',
    'Fabiola', 'Fae', 'Fairy', 'Faith', 'Fallon', 'Falon', 'Fannie',
    'Fanny', 'Fannye', 'Farah', 'Farrah', 'Fatima', 'Fawn', 'Fay', 'Faye',
    'Felecia', 'Felice', 'Felicia', 'Felicie', 'Felicitas', 'Felicity',
    'Felipa', 'Felisha', 'Fern', 'Fernanda', 'Ferne', 'Fidelia', 'Filomena',
    'Finley', 'Fiona', 'Flavia', 'Fleda', 'Fleeta', 'Fleta', 'Flo',
    'Flonnie', 'Flor', 'Flora', 'Florance', 'Florence', 'Florene',
    'Floretta', 'Florida', 'Florie', 'Florine', 'Florrie', 'Flossie',
    'Floy', 'Fonda', 'Forest', 'Fran', 'Franc', 'Frances', 'Francesca',
    'Francies', 'Francina', 'Francine', 'Francis', 'Francisca',
    'Francisquita', 'Frankie', 'Freda', 'Freddie', 'Frederica',
    'Fredericka', 'Freeda', 'Freida', 'Frida', 'Frieda', 'Frona', 'Fronia',
    'Fronie', 'Fronnie', 'Fumiko', 'Gabriela', 'Gabriella', 'Gabrielle',
    'Gail', 'Gale', 'Galilea', 'Garnet', 'Garnett', 'Gay', 'Gaye', 'Gayla',
    'Gayle', 'Gaylene', 'Gaynell', 'Gearldine', 'Gemma', 'Gena', 'Gene',
    'Genesis', 'Geneva', 'Genevieve', 'Genevra', 'Genie', 'Gennie',
    'Genoveva', 'Georganna', 'Georgeann', 'Georgeanna', 'Georgene',
    'Georgetta', 'Georgette', 'Georgia', 'Georgiana', 'Georgiann',
    'Georgianna', 'Georgie', 'Georgina', 'Georgine', 'Geraldine', 'Geralyn',
    'Gerda', 'Geri', 'Germaine', 'Gerri', 'Gerry', 'Gertha', 'Gertie',
    'Gertrude', 'Gia', 'Giada', 'Giana', 'Gianna', 'Gidget', 'Gigi',
    'Gilda', 'Gillian', 'Gillie', 'Gina', 'Ginger', 'Ginny', 'Giovanna',
    'Girtha', 'Gisele', 'Giselle', 'Gisselle', 'Giuliana', 'Gladis',
    'Gladyce', 'Gladys', 'Glenda', 'Glendora', 'Glenn', 'Glenna', 'Glennie',
    'Glennis', 'Glinda', 'Gloria', 'Glynda', 'Glynis', 'Golda', 'Golden',
    'Goldia', 'Goldie', 'Grace', 'Gracelyn', 'Gracia', 'Gracie', 'Graciela',
    'Grayce', 'Grecia', 'Gregoria', 'Greta', 'Gretchen', 'Gretta', 'Grisel',
    'Griselda', 'Guadalupe', 'Gunda', 'Gussie', 'Gusta', 'Gustie', 'Gwen',
    'Gwenda', 'Gwendolyn', 'Gwyn', 'Gwyneth', 'Hadassah', 'Hadley',
    'Hailee', 'Hailey', 'Hailie', 'Haleigh', 'Haley', 'Hali', 'Halie',
    'Halle', 'Halley', 'Hallie', 'Hana', 'Hanna', 'Hannah', 'Harlene',
    'Harley', 'Harlow', 'Harmony', 'Harper', 'Harriet', 'Harriett',
    'Harriette', 'Haruko', 'Hasel', 'Hassie', 'Hattie', 'Haven', 'Hayden',
    'Haylee', 'Hayleigh', 'Hayley', 'Haylie', 'Hazel', 'Hazelle', 'Hazle',
    'Heather', 'Heaven', 'Hedwig', 'Hedy', 'Heidi', 'Heidy', 'Helaine',
    'Helen', 'Helena', 'Helene', 'Helga', 'Hellen', 'Helma', 'Helyn',
    'Hennie', 'Henretta', 'Henrietta', 'Henriette', 'Herlinda', 'Herma',
    'Hermina', 'Hermine', 'Herminia', 'Hertha', 'Hessie', 'Hester',
    'Hettie', 'Hetty', 'Hilah', 'Hilary', 'Hilda', 'Hildegard',
    'Hildegarde', 'Hildred', 'Hildur', 'Hillary', 'Hilma', 'Holli',
    'Hollie', 'Hollis', 'Holly', 'Honora', 'Hope', 'Hortencia', 'Hortense',
    'Hortensia', 'Hulda', 'Huldah', 'Hunter', 'Ica', 'Icey', 'Icie', 'Icy',
    'Ida', 'Idabelle', 'Idamae', 'Idell', 'Idella', 'Iesha', 'Ieshia',
    'Ila', 'Ilah', 'Ilda', 'Ilene', 'Iliana', 'Illa', 'Ilma', 'Ilo',
    'Ilona', 'Ima', 'Imani', 'Imelda', 'Imo', 'Imogene', 'Ina', 'India',
    'Indiana', 'Inell', 'Ines', 'Inez', 'Infant', 'Inga', 'Ingeborg',
    'Inger', 'Ingrid', 'Iola', 'Iona', 'Ione', 'Ira', 'Ireland', 'Irena',
    'Irene', 'Iridian', 'Irine', 'Iris', 'Irma', 'Irva', 'Isa', 'Isabel',
    'Isabela', 'Isabell', 'Isabella', 'Isabelle', 'Isadora', 'Isamar',
    'Isis', 'Isla', 'Isobel', 'Itzel', 'Iva', 'Ivah', 'Ivana', 'Ivanna',
    'Ivette', 'Ivey', 'Ivie', 'Ivonne', 'Ivory', 'Ivy', 'Iyana', 'Iyanna',
    'Iza', 'Izabella', 'Izabelle', 'Izetta', 'Izola', 'Izora', 'Jacalyn',
    'Jacey', 'Jackeline', 'Jacki', 'Jackie', 'Jacklyn', 'Jaclyn', 'Jacque',
    'Jacquelin', 'Jacqueline', 'Jacquelyn', 'Jacquline', 'Jacqulyn', 'Jada',
    'Jade', 'Jaden', 'Jadyn', 'Jaeda', 'Jaelyn', 'Jaelynn', 'Jaida',
    'Jaiden', 'Jaidyn', 'Jailene', 'Jailyn', 'Jaime', 'Jaimee', 'Jakayla',
    'Jaleesa', 'Jalisa', 'Jalissa', 'Jaliyah', 'Jalyn', 'Jalynn', 'Jamey',
    'Jami', 'Jamie', 'Jamila', 'Jamiya', 'Jammie', 'Jamya', 'Jan', 'Jana',
    'Janae', 'Janay', 'Jane', 'Janeen', 'Janel', 'Janell', 'Janelle',
    'Janene', 'Janessa', 'Janet', 'Janette', 'Janey', 'Janiah', 'Janice',
    'Janie', 'Janine', 'Janis', 'Janiya', 'Janiyah', 'Jann', 'Janna',
    'Jannette', 'Jannie', 'January', 'Janyce', 'Jaquelin', 'Jaqueline',
    'Jaslene', 'Jaslyn', 'Jasmin', 'Jasmine', 'Jasmyn', 'Jasmyne',
    'Jaunita', 'Jaycee', 'Jaycie', 'Jayda', 'Jayde', 'Jayden', 'Jaye',
    'Jayla', 'Jaylah', 'Jaylee', 'Jayleen', 'Jaylen', 'Jaylene', 'Jaylin',
    'Jaylyn', 'Jaylynn', 'Jayme', 'Jayne', 'Jazlene', 'Jazlyn', 'Jazlynn',
    'Jazmin', 'Jazmine', 'Jazmyn', 'Jazmyne', 'Jean', 'Jeana', 'Jeane',
    'Jeanetta', 'Jeanette', 'Jeanie', 'Jeanine', 'Jeanmarie', 'Jeanna',
    'Jeanne', 'Jeannette', 'Jeannie', 'Jeannine', 'Jeffie', 'Jemima',
    'Jena', 'Jenelle', 'Jenifer', 'Jenilee', 'Jenna', 'Jennette', 'Jenni',
    'Jennie', 'Jennifer', 'Jenniffer', 'Jenny', 'Jensen', 'Jeraldine',
    'Jeri', 'Jerica', 'Jerilyn', 'Jerilynn', 'Jerri', 'Jerrica', 'Jerrie',
    'Jerrilyn', 'Jerusha', 'Jeryl', 'Jesenia', 'Jesica', 'Jesse',
    'Jessenia', 'Jessi', 'Jessica', 'Jessie', 'Jessika', 'Jessye', 'Jetta',
    'Jettie', 'Jewel', 'Jewell', 'Jill', 'Jillian', 'Jimena', 'Jinnie',
    'Jo', 'Joan', 'Joana', 'Joanie', 'Joann', 'Joanna', 'Joanne', 'Jocelyn',
    'Jocelyne', 'Jocelynn', 'Jodi', 'Jodie', 'Jody', 'Joell', 'Joella',
    'Joelle', 'Joellen', 'Joetta', 'Joette', 'Johana', 'Johanna',
    'Johannah', 'Johnie', 'Johnna', 'Johnnie', 'Joi', 'Joleen', 'Jolene',
    'Jolette', 'Jolie', 'Joline', 'Jonell', 'Joni', 'Jonna', 'Jonnie',
    'Jordan', 'Jordin', 'Jordyn', 'Joretta', 'Jorja', 'Josefa', 'Josefina',
    'Josefita', 'Joselin', 'Joseline', 'Joselyn', 'Josephine', 'Josette',
    'Josie', 'Josiephine', 'Joslyn', 'Jossie', 'Journey', 'Jovita', 'Joy',
    'Joyce', 'Joycelyn', 'Joye', 'Juana', 'Juanita', 'Judi', 'Judie',
    'Judith', 'Judy', 'Judyth', 'Jule', 'Juli', 'Julia', 'Juliana',
    'Juliann', 'Julianna', 'Julianne', 'Julie', 'Juliet', 'Juliette',
    'Julisa', 'Julissa', 'June', 'Junia', 'Junie', 'Justice', 'Justina',
    'Justine', 'Kaaren', 'Kacey', 'Kaci', 'Kacie', 'Kacy', 'Kadence',
    'Kadijah', 'Kaela', 'Kaelyn', 'Kaelynn', 'Kaia', 'Kaila', 'Kailee',
    'Kailey', 'Kailyn', 'Kaitlin', 'Kaitlyn', 'Kaitlynn', 'Kaiya', 'Kala',
    'Kaleena', 'Kaleigh', 'Kalene', 'Kaley', 'Kali', 'Kalie', 'Kaliyah',
    'Kallie', 'Kalyn', 'Kamari', 'Kameron', 'Kami', 'Kamila', 'Kamilah',
    'Kamora', 'Kamryn', 'Kamya', 'Kandace', 'Kandi', 'Kandice', 'Kandy',
    'Kanesha', 'Kanisha', 'Kara', 'Karan', 'Karel', 'Karen', 'Kari',
    'Karie', 'Karin', 'Karina', 'Karis', 'Karissa', 'Karla', 'Karlee',
    'Karlene', 'Karley', 'Karli', 'Karlie', 'Karly', 'Karma', 'Karol',
    'Karolyn', 'Karon', 'Karren', 'Karri', 'Karrie', 'Karsyn', 'Karyl',
    'Karyme', 'Karyn', 'Kasandra', 'Kasey', 'Kasie', 'Kassandra', 'Kassidy',
    'Kassie', 'Katarina', 'Kate', 'Katelin', 'Katelyn', 'Katelynn',
    'Katerina', 'Kathaleen', 'Katharina', 'Katharine', 'Katharyn',
    'Katherin', 'Katherine', 'Kathern', 'Katheryn', 'Kathey', 'Kathi',
    'Kathie', 'Kathleen', 'Kathlene', 'Kathlyn', 'Kathrine', 'Kathryn',
    'Kathryne', 'Kathy', 'Kathyrn', 'Kati', 'Katia', 'Katie', 'Katina',
    'Katlin', 'Katlyn', 'Katlynn', 'Katrina', 'Kattie', 'Katy', 'Kay',
    'Kaya', 'Kaycee', 'Kayden', 'Kaydence', 'Kaye', 'Kayla', 'Kaylah',
    'Kaylan', 'Kaylee', 'Kayleen', 'Kayleigh', 'Kaylen', 'Kaylene',
    'Kayley', 'Kayli', 'Kaylie', 'Kaylin', 'Kaylyn', 'Kaylynn', 'Kazuko',
    'Keanna', 'Keara', 'Kecia', 'Keeley', 'Keely', 'Keena', 'Keesha',
    'Keila', 'Keira', 'Keisha', 'Kelcie', 'Keli', 'Kelis', 'Kellee',
    'Kelley', 'Kelli', 'Kellie', 'Kelly', 'Kelsea', 'Kelsey', 'Kelsi',
    'Kelsie', 'Kendal', 'Kendall', 'Kendra', 'Kenia', 'Kenisha', 'Kenley',
    'Kenna', 'Kennedi', 'Kennedy', 'Kenya', 'Kenyatta', 'Kenzie', 'Keri',
    'Kerri', 'Kerrie', 'Kerry', 'Kesha', 'Keshia', 'Keyla', 'Khadijah',
    'Khalilah', 'Khloe', 'Kia', 'Kiana', 'Kianna', 'Kiara', 'Kiarra',
    'Kiera', 'Kierra', 'Kiersten', 'Kiley', 'Kim', 'Kimber', 'Kimberely',
    'Kimberlee', 'Kimberley', 'Kimberli', 'Kimberlie', 'Kimberly', 'Kimora',
    'Kindra', 'Kinley', 'Kinsey', 'Kinsley', 'Kira', 'Kirsten', 'Kirstie',
    'Kirstin', 'Kisha', 'Kittie', 'Kitty', 'Kiya', 'Kiyoko', 'Kizzie',
    'Kizzy', 'Kloe', 'Kori', 'Kortney', 'Kourtney', 'Kris', 'Krissy',
    'Krista', 'Kristal', 'Kristan', 'Kristen', 'Kristi', 'Kristian',
    'Kristie', 'Kristin', 'Kristina', 'Kristine', 'Kristy', 'Kristyn',
    'Krysta', 'Krystal', 'Krysten', 'Krystin', 'Krystina', 'Krystle', 'Kya',
    'Kyara', 'Kyla', 'Kylah', 'Kyle', 'Kylee', 'Kyleigh', 'Kylene', 'Kylie',
    'Kyra', 'Kyrie', 'Lacey', 'Laci', 'Lacie', 'Lacy', 'Ladonna', 'Lady',
    'Lahoma', 'Laila', 'Lailah', 'Lainey', 'Laisha', 'Lakeisha', 'Laken',
    'Lakendra', 'Lakesha', 'Lakeshia', 'Lakisha', 'Lala', 'Lalla', 'Lana',
    'Lanette', 'Laney', 'Lani', 'Lanie', 'Lanita', 'Lannie', 'Laquita',
    'Lara', 'Larae', 'Laraine', 'Larissa', 'Larue', 'Lashanda', 'Lashawn',
    'Lashonda', 'Lashunda', 'Lasonya', 'Lassie', 'Latanya', 'Latarsha',
    'Latasha', 'Latesha', 'Latifah', 'Latisha', 'Latonia', 'Latonya',
    'Latoria', 'Latosha', 'Latoya', 'Latoyia', 'Latrice', 'Latricia',
    'Latrina', 'Launa', 'Laura', 'Laureen', 'Laurel', 'Lauren', 'Laurene',
    'Lauretta', 'Laurette', 'Lauri', 'Laurie', 'Laurine', 'Lauryn',
    'Lavada', 'Lavelle', 'Lavenia', 'Lavera', 'Lavern', 'Laverna',
    'Laverne', 'Lavina', 'Lavinia', 'Lavon', 'Lavona', 'Lavonda', 'Lavonia',
    'Lavonne', 'Lawanda', 'Layla', 'Laylah', 'Lea', 'Leafy', 'Leah',
    'Leala', 'Leana', 'Leandra', 'Leaner', 'Leann', 'Leanna', 'Leanne',
    'Leatha', 'Leatrice', 'Leda', 'Lee', 'Leeann', 'Leesa', 'Leia', 'Leigh',
    'Leighton', 'Leila', 'Leilani', 'Leisa', 'Leisha', 'Leitha', 'Lela',
    'Lelah', 'Lelar', 'Lelia', 'Lella', 'Lemma', 'Lempi', 'Lena', 'Lenna',
    'Lennie', 'Lenora', 'Lenore', 'Leola', 'Leoma', 'Leona', 'Leone',
    'Leonia', 'Leonie', 'Leonor', 'Leonora', 'Leonore', 'Leontine', 'Leora',
    'Leota', 'Lera', 'Lesa', 'Lesia', 'Leslee', 'Lesley', 'Lesli', 'Leslie',
    'Lesly', 'Lessie', 'Lesta', 'Leta', 'Letha', 'Lethia', 'Leticia',
    'Letitia', 'Letta', 'Lettie', 'Letty', 'Leva', 'Levina', 'Lexi',
    'Lexie', 'Lexis', 'Lexus', 'Leyla', 'Lia', 'Liana', 'Liane', 'Libbie',
    'Libby', 'Liberty', 'Lida', 'Liddie', 'Lidia', 'Lidie', 'Lila', 'Lilah',
    'Lilia', 'Lilian', 'Liliana', 'Lilianna', 'Lilie', 'Lilla', 'Liller',
    'Lillia', 'Lillian', 'Lilliana', 'Lillianna', 'Lillie', 'Lillis',
    'Lilly', 'Lily', 'Lilyan', 'Lilyana', 'Lilyanna', 'Lina', 'Linda',
    'Lindsay', 'Lindsey', 'Lindy', 'Linette', 'Linna', 'Linnea', 'Linnie',
    'Linsey', 'Lisa', 'Lisbeth', 'Lise', 'Lisette', 'Lisha', 'Lissa',
    'Lissette', 'Lissie', 'Lita', 'Litha', 'Littie', 'Litzy', 'Livia',
    'Liz', 'Liza', 'Lizabeth', 'Lizbeth', 'Lizeth', 'Lizette', 'Lizzie',
    'Lockie', 'Loda', 'Logan', 'Lois', 'Lola', 'Lolita', 'Lolla', 'Lollie',
    'Loma', 'Lona', 'London', 'Londyn', 'Loni', 'Lonie', 'Lonna', 'Lonnie',
    'Lora', 'Loraine', 'Lorayne', 'Lorean', 'Loree', 'Loreen', 'Lorelai',
    'Lorelei', 'Loren', 'Lorena', 'Lorene', 'Lorenza', 'Loretta', 'Loretto',
    'Lori', 'Loria', 'Loriann', 'Lorie', 'Lorinda', 'Lorine', 'Loris',
    'Lorna', 'Lorraine', 'Lorrayne', 'Lorri', 'Lorrie', 'Lossie', 'Lota',
    'Lotta', 'Lottie', 'Lou', 'Louann', 'Louanna', 'Louella', 'Louetta',
    'Louie', 'Louisa', 'Louise', 'Louisiana', 'Loula', 'Lourdes',
    'Louvenia', 'Love', 'Lovey', 'Lovie', 'Lovina', 'Lovisa', 'Loyce', 'Lu',
    'Luana', 'Luann', 'Luanne', 'Luberta', 'Lucero', 'Lucetta', 'Lucia',
    'Luciana', 'Lucie', 'Lucile', 'Lucille', 'Lucina', 'Lucinda', 'Lucindy',
    'Lucretia', 'Lucy', 'Luda', 'Ludie', 'Lue', 'Luella', 'Luetta',
    'Lugenia', 'Luisa', 'Lula', 'Lulah', 'Lular', 'Lulie', 'Lulla', 'Lulu',
    'Luna', 'Lupe', 'Lura', 'Lurana', 'Lurena', 'Lurline', 'Lutie',
    'Luvenia', 'Luverne', 'Luvinia', 'Luz', 'Lyda', 'Lydia', 'Lyla',
    'Lylah', 'Lyn', 'Lynda', 'Lyndia', 'Lyndsay', 'Lyndsey', 'Lynette',
    'Lynn', 'Lynne', 'Lynnette', 'Lynsey', 'Lyric', 'Mabel', 'Mabell',
    'Mabelle', 'Mable', 'Macel', 'Macey', 'Machelle', 'Maci', 'Macie',
    'Mackenzie', 'Macy', 'Madaline', 'Madalyn', 'Madalynn', 'Maddison',
    'Madeleine', 'Madelene', 'Madeline', 'Madelyn', 'Madelynn', 'Madge',
    'Madie', 'Madilyn', 'Madilynn', 'Madisen', 'Madison', 'Madisyn',
    'Madlyn', 'Madonna', 'Madora', 'Madyson', 'Mae', 'Maebell', 'Maebelle',
    'Maegan', 'Maeve', 'Mafalda', 'Magan', 'Magdalen', 'Magdalena',
    'Magdalene', 'Magen', 'Maggie', 'Magnolia', 'Mahala', 'Mahalia',
    'Mahalie', 'Mai', 'Maia', 'Maida', 'Maira', 'Maiya', 'Makaila',
    'Makala', 'Makayla', 'Makena', 'Makenna', 'Makenzie', 'Malaya',
    'Maleah', 'Malia', 'Maliah', 'Malinda', 'Malissa', 'Malissie',
    'Maliyah', 'Mallie', 'Mallorie', 'Mallory', 'Malorie', 'Malvina',
    'Mame', 'Mamie', 'Mammie', 'Manda', 'Mandi', 'Mandie', 'Mandy',
    'Manerva', 'Manervia', 'Manie', 'Manila', 'Manilla', 'Mannie',
    'Manuela', 'Manuelita', 'Mara', 'Maralyn', 'Maranda', 'Marcela',
    'Marcelina', 'Marceline', 'Marcella', 'Marcelle', 'Marci', 'Marcia',
    'Marcie', 'Marcy', 'Mardell', 'Mareli', 'Marely', 'Maren', 'Margaret',
    'Margarete', 'Margaretha', 'Margarett', 'Margaretta', 'Margarette',
    'Margarita', 'Margarite', 'Marge', 'Margene', 'Margeret', 'Margery',
    'Marget', 'Margie', 'Margo', 'Margot', 'Margret', 'Margrett',
    'Margretta', 'Marguerite', 'Margueritte', 'Margurite', 'Margy', 'Mari',
    'Maria', 'Mariah', 'Mariam', 'Marian', 'Mariana', 'Marianita',
    'Mariann', 'Marianna', 'Marianne', 'Maribel', 'Maribeth', 'Maricela',
    'Marie', 'Mariel', 'Mariela', 'Marietta', 'Marilee', 'Marilla',
    'Marilou', 'Marilyn', 'Marilynn', 'Marin', 'Marina', 'Marinda',
    'Marion', 'Marisa', 'Marisela', 'Marisol', 'Marissa', 'Marita',
    'Maritza', 'Mariyah', 'Marjorie', 'Marjory', 'Markita', 'Marla',
    'Marlana', 'Marlee', 'Marleen', 'Marleigh', 'Marlen', 'Marlena',
    'Marlene', 'Marley', 'Marlie', 'Marlo', 'Marlyn', 'Marlys', 'Marni',
    'Marnie', 'Marnita', 'Marolyn', 'Marquita', 'Marry', 'Marsha', 'Marta',
    'Martha', 'Marti', 'Martika', 'Martina', 'Martine', 'Marty', 'Marva',
    'Marvel', 'Mary', 'Maryam', 'Maryann', 'Maryanne', 'Marybelle',
    'Marybeth', 'Maryellen', 'Maryjane', 'Maryjo', 'Marylee', 'Marylin',
    'Marylou', 'Marylouise', 'Marylyn', 'Masako', 'Mathilda', 'Mathilde',
    'Matie', 'Matilda', 'Matilde', 'Mattie', 'Mattye', 'Maud', 'Maude',
    'Maudie', 'Maura', 'Maureen', 'Maurine', 'Mavis', 'Maxie', 'Maxine',
    'May', 'Maya', 'Maybell', 'Maybelle', 'Maye', 'Mayme', 'Maymie',
    'Mayra', 'Mazie', 'Mckayla', 'Mckenna', 'Mckenzie', 'Mckinley',
    'Meadow', 'Meagan', 'Meaghan', 'Mechelle', 'Meda', 'Media', 'Medora',
    'Meg', 'Megan', 'Meggan', 'Meghan', 'Meghann', 'Melanie', 'Melany',
    'Melba', 'Melina', 'Melinda', 'Melisa', 'Melissa', 'Melissia', 'Mell',
    'Mellie', 'Mellisa', 'Mellissa', 'Melodee', 'Melodie', 'Melody',
    'Melonie', 'Melony', 'Melva', 'Melvina', 'Mena', 'Mendy', 'Mercedes',
    'Mercy', 'Meredith', 'Merilyn', 'Merle', 'Merlene', 'Merna', 'Merri',
    'Merrie', 'Merrilee', 'Merrily', 'Merry', 'Mertie', 'Meryl', 'Meta',
    'Metha', 'Metta', 'Mettie', 'Mia', 'Miah', 'Micaela', 'Micah',
    'Micayla', 'Michaela', 'Michaele', 'Michal', 'Michele', 'Michelina',
    'Michell', 'Michelle', 'Mickey', 'Mickie', 'Miesha', 'Migdalia',
    'Mignon', 'Mikaela', 'Mikaila', 'Mikala', 'Mikalah', 'Mikayla', 'Mila',
    'Milagros', 'Milan', 'Milda', 'Mildred', 'Miley', 'Milissa',
    'Millicent', 'Millie', 'Milly', 'Mima', 'Mimi', 'Mina', 'Minda',
    'Mindi', 'Mindy', 'Minerva', 'Minervia', 'Minna', 'Minnie', 'Minta',
    'Mintie', 'Mira', 'Miracle', 'Miranda', 'Mireya', 'Miriah', 'Miriam',
    'Mirna', 'Mirtie', 'Missie', 'Missouri', 'Missy', 'Misti', 'Mistie',
    'Misty', 'Mittie', 'Mitzi', 'Miya', 'Modena', 'Moesha', 'Moira',
    'Mollie', 'Molly', 'Mona', 'Monica', 'Monika', 'Monique', 'Monna',
    'Monnie', 'Monserrat', 'Montana', 'Montie', 'Mora', 'Morgan', 'Moriah',
    'Mossie', 'Mozell', 'Mozella', 'Mozelle', 'Muriel', 'Murl', 'Mya',
    'Myah', 'Myla', 'Mylee', 'Mylie', 'Myra', 'Myranda', 'Myrl', 'Myrle',
    'Myrna', 'Myrta', 'Myrtice', 'Myrtie', 'Myrtis', 'Myrtle', 'Nada',
    'Nadia', 'Nadine', 'Naima', 'Nakia', 'Nakisha', 'Nakita', 'Nallely',
    'Nan', 'Nana', 'Nanci', 'Nancie', 'Nancy', 'Nanette', 'Nanie', 'Nanna',
    'Nannette', 'Nannie', 'Naoma', 'Naomi', 'Narcissus', 'Natalee',
    'Natalia', 'Natalie', 'Nataly', 'Natalya', 'Natasha', 'Nathalia',
    'Nathalie', 'Nathaly', 'Natosha', 'Nautica', 'Nayeli', 'Nayely',
    'Nealie', 'Nealy', 'Nedra', 'Neha', 'Nelda', 'Nelia', 'Nelie', 'Nell',
    'Nella', 'Nelle', 'Nellie', 'Nelly', 'Nena', 'Neola', 'Neoma', 'Neppie',
    'Nereida', 'Neta', 'Netta', 'Nettie', 'Neva', 'Nevada', 'Nevaeh',
    'Neveah', 'Nia', 'Nichelle', 'Nichol', 'Nichole', 'Nicki', 'Nicola',
    'Nicole', 'Nicolette', 'Nicolle', 'Niki', 'Nikia', 'Nikita', 'Nikki',
    'Nikole', 'Nila', 'Nilda', 'Nina', 'Ninnie', 'Nira', 'Nita', 'Nobie',
    'Noel', 'Noelia', 'Noelle', 'Noemi', 'Noemie', 'Nohely', 'Nola',
    'Nolia', 'Nolie', 'Noma', 'Nona', 'Nonie', 'Nora', 'Norah', 'Noreen',
    'Norene', 'Noreta', 'Noretta', 'Norine', 'Norita', 'Norma', 'Nova',
    'Novella', 'Nya', 'Nyah', 'Nyasia', 'Nyla', 'Nylah', 'Nyree', 'Ocie',
    'Octa', 'Octavia', 'Octavie', 'Oda', 'Odalis', 'Odalys', 'Odelia',
    'Odell', 'Odessa', 'Odette', 'Odie', 'Odile', 'Ofelia', 'Ola', 'Olar',
    'Olena', 'Olene', 'Oleta', 'Olevia', 'Olga', 'Olie', 'Olinda', 'Oline',
    'Oliva', 'Olive', 'Olivia', 'Olivine', 'Ollie', 'Olympia', 'Oma',
    'Omie', 'Ona', 'Oneida', 'Oneta', 'Oney', 'Onie', 'Onnie', 'Opal',
    'Opha', 'Ophelia', 'Ora', 'Orah', 'Oral', 'Oralia', 'Orelia', 'Orene',
    'Orilla', 'Orlena', 'Orma', 'Orpha', 'Orra', 'Orrie', 'Osa', 'Osie',
    'Ossie', 'Ota', 'Otelia', 'Otha', 'Ottie', 'Ottilia', 'Ottilie',
    'Ouida', 'Ova', 'Ozell', 'Ozella', 'Ozie', 'Paige', 'Pairlee',
    'Paisley', 'Paityn', 'Pallie', 'Palma', 'Paloma', 'Pam', 'Pamala',
    'Pamela', 'Pamelia', 'Pamella', 'Pandora', 'Pansy', 'Paola', 'Paralee',
    'Paris', 'Parker', 'Parlee', 'Parthenia', 'Pat', 'Patience', 'Patrica',
    'Patrice', 'Patricia', 'Patsy', 'Patti', 'Pattie', 'Patty', 'Paula',
    'Pauletta', 'Paulette', 'Paulina', 'Pauline', 'Payten', 'Payton',
    'Pearl', 'Pearla', 'Pearle', 'Pearlene', 'Pearlie', 'Pearline',
    'Pearly', 'Peggie', 'Peggy', 'Penelope', 'Penni', 'Pennie', 'Penny',
    'Pepper', 'Perla', 'Permelia', 'Perri', 'Petra', 'Peyton', 'Phebe',
    'Pheobe', 'Phillis', 'Philomena', 'Philomene', 'Phoebe', 'Phoenix',
    'Phylicia', 'Phylis', 'Phyliss', 'Phyllis', 'Pink', 'Pinkey', 'Pinkie',
    'Piper', 'Pluma', 'Pollie', 'Polly', 'Porsche', 'Porsha', 'Portia',
    'Precious', 'Presley', 'Pricilla', 'Princess', 'Priscila', 'Priscilla',
    'Prudence', 'Prudie', 'Qiana', 'Queen', 'Queenie', 'Quiana', 'Quinn',
    'Rachael', 'Racheal', 'Rachel', 'Rachelle', 'Racquel', 'Rae', 'Raegan',
    'Raelyn', 'Raelynn', 'Rafaela', 'Ragna', 'Raina', 'Ramona', 'Randi',
    'Raquel', 'Rashida', 'Raven', 'Rayna', 'Rayne', 'Reagan', 'Reanna',
    'Reatha', 'Reba', 'Rebeca', 'Rebecca', 'Rebekah', 'Reece', 'Reese',
    'Regan', 'Regena', 'Regenia', 'Regina', 'Reilly', 'Reina', 'Rella',
    'Rena', 'Renada', 'Renae', 'Renata', 'Rene', 'Renea', 'Renee', 'Renita',
    'Rennie', 'Ressie', 'Reta', 'Retha', 'Retta', 'Rettie', 'Reva', 'Reyna',
    'Rhea', 'Rheta', 'Rhianna', 'Rhiannon', 'Rhoda', 'Rhona', 'Rhonda',
    'Rianna', 'Richelle', 'Ricki', 'Rihanna', 'Rikki', 'Riley', 'Rilla',
    'Rillie', 'Rinda', 'Risa', 'Rita', 'River', 'Riya', 'Robbie', 'Robbin',
    'Roberta', 'Robin', 'Robyn', 'Rochelle', 'Rocio', 'Roena', 'Rolanda',
    'Roma', 'Romaine', 'Romona', 'Rona', 'Ronda', 'Roni', 'Ronna', 'Ronnie',
    'Rory', 'Rosa', 'Rosabelle', 'Rosalee', 'Rosalia', 'Rosalie',
    'Rosalind', 'Rosalinda', 'Rosaline', 'Rosalyn', 'Rosamond', 'Rosann',
    'Rosanna', 'Rosanne', 'Rosaria', 'Rosario', 'Rose', 'Roseann',
    'Roseanna', 'Roseanne', 'Rosella', 'Roselyn', 'Rosemarie', 'Rosemary',
    'Rosena', 'Rosetta', 'Rosey', 'Rosia', 'Rosie', 'Rosina', 'Rosita',
    'Roslyn', 'Rossie', 'Rosy', 'Rowan', 'Rowena', 'Roxana', 'Roxane',
    'Roxann', 'Roxanna', 'Roxanne', 'Roxie', 'Roxy', 'Rozanne', 'Rozella',
    'Rubi', 'Rubie', 'Ruby', 'Rubye', 'Ruie', 'Ruth', 'Rutha', 'Ruthann',
    'Ruthanne', 'Ruthe', 'Ruthie', 'Ryann', 'Rylan', 'Rylee', 'Ryleigh',
    'Rylie', 'Sabina', 'Sable', 'Sabra', 'Sabrina', 'Sada', 'Sade', 'Sadie',
    'Sadye', 'Sage', 'Saige', 'Salena', 'Salina', 'Sallie', 'Sally',
    'Salma', 'Salome', 'Samantha', 'Samara', 'Samatha', 'Samira', 'Samiyah',
    'Sammie', 'Sanaa', 'Sanai', 'Sandi', 'Sandie', 'Sandra', 'Sandy',
    'Saniya', 'Saniyah', 'Sanjuana', 'Sanjuanita', 'Sannie', 'Santa',
    'Santana', 'Santina', 'Santos', 'Sara', 'Sarah', 'Sarahi', 'Sarai',
    'Sariah', 'Sarina', 'Sarita', 'Sarrah', 'Sasha', 'Saundra', 'Savana',
    'Savanah', 'Savanna', 'Savannah', 'Savilla', 'Scarlet', 'Scarlett',
    'Sebrina', 'Selah', 'Selena', 'Selene', 'Selina', 'Selma', 'Sena',
    'Senora', 'Serena', 'Serenity', 'Serina', 'Shae', 'Shaina', 'Shakira',
    'Shalon', 'Shalonda', 'Shameka', 'Shamika', 'Shana', 'Shanae', 'Shanda',
    'Shandra', 'Shane', 'Shaneka', 'Shanell', 'Shanelle', 'Shanequa',
    'Shani', 'Shania', 'Shanice', 'Shaniece', 'Shanika', 'Shaniqua',
    'Shanita', 'Shaniya', 'Shanna', 'Shannan', 'Shannen', 'Shannon',
    'Shanon', 'Shanta', 'Shante', 'Shantel', 'Shantell', 'Shaquana',
    'Shaquita', 'Shara', 'Shardae', 'Sharday', 'Sharde', 'Sharee', 'Sharen',
    'Shari', 'Sharita', 'Sharla', 'Sharleen', 'Sharlene', 'Sharman',
    'Sharon', 'Sharonda', 'Sharron', 'Sharyl', 'Sharyn', 'Shasta',
    'Shatara', 'Shauna', 'Shaunna', 'Shavon', 'Shavonne', 'Shawanda',
    'Shawna', 'Shawnda', 'Shawnee', 'Shawnna', 'Shawnte', 'Shay', 'Shayla',
    'Shaylee', 'Shayna', 'Shea', 'Sheena', 'Sheila', 'Sheilah', 'Shelba',
    'Shelbi', 'Shelbie', 'Shelby', 'Shelia', 'Shelley', 'Shelli', 'Shellie',
    'Shelly', 'Shelva', 'Shelvia', 'Shelvie', 'Shena', 'Shenna', 'Sheree',
    'Sheri', 'Sheridan', 'Sherie', 'Sherilyn', 'Sherita', 'Sherlyn',
    'Sheron', 'Sherree', 'Sherri', 'Sherrie', 'Sherrill', 'Sherron',
    'Sherry', 'Sherryl', 'Sheryl', 'Sheryll', 'Sheyla', 'Shianne', 'Shiela',
    'Shiloh', 'Shira', 'Shirl', 'Shirlee', 'Shirleen', 'Shirlene',
    'Shirley', 'Shirleyann', 'Shirlie', 'Shona', 'Shonda', 'Shonna',
    'Shreya', 'Shyann', 'Shyanne', 'Shyla', 'Sibbie', 'Sibyl', 'Siddie',
    'Sidney', 'Siena', 'Sienna', 'Sierra', 'Signa', 'Signe', 'Sigrid',
    'Silvia', 'Simona', 'Simone', 'Sina', 'Sinda', 'Siobhan', 'Sister',
    'Sky', 'Skye', 'Skyla', 'Skylar', 'Skyler', 'Sloane', 'Socorro',
    'Sofia', 'Soledad', 'Somer', 'Sommer', 'Sondra', 'Sonia', 'Sonja',
    'Sonji', 'Sonya', 'Sophia', 'Sophie', 'Sophronia', 'Spring', 'Stacey',
    'Staci', 'Stacia', 'Stacie', 'Stacy', 'Star', 'Starla', 'Starr',
    'Stasia', 'Stefani', 'Stefanie', 'Stella', 'Stephaine', 'Stephani',
    'Stephania', 'Stephanie', 'Stephany', 'Stephenie', 'Stevie', 'Stormy',
    'Sudie', 'Sue', 'Suellen', 'Sula', 'Summer', 'Sunday', 'Sunny',
    'Sunshine', 'Susan', 'Susana', 'Susann', 'Susanna', 'Susannah',
    'Susanne', 'Susie', 'Sussie', 'Suzan', 'Suzann', 'Suzanna', 'Suzanne',
    'Suzette', 'Suzie', 'Suzy', 'Sybil', 'Sybilla', 'Syble', 'Sydell',
    'Sydnee', 'Sydney', 'Sydni', 'Sydnie', 'Sylva', 'Sylvania', 'Sylvia',
    'Symone', 'Syreeta', 'Tabatha', 'Tabetha', 'Tabitha', 'Tai', 'Taina',
    'Taja', 'Takisha', 'Talia', 'Taliyah', 'Tamala', 'Tamara', 'Tamatha',
    'Tambra', 'Tameka', 'Tamekia', 'Tamela', 'Tamera', 'Tami', 'Tamia',
    'Tamica', 'Tamie', 'Tamika', 'Tamiko', 'Tamisha', 'Tammi', 'Tammie',
    'Tammy', 'Tamra', 'Tamya', 'Tana', 'Tanesha', 'Tangela', 'Tania',
    'Tanika', 'Tanisha', 'Taniya', 'Taniyah', 'Tanja', 'Tanya', 'Tara',
    'Tarah', 'Taraji', 'Tari', 'Tarsha', 'Taryn', 'Tasha', 'Tashina',
    'Tasia', 'Tatia', 'Tatiana', 'Tatianna', 'Tatum', 'Tatyana', 'Tatyanna',
    'Tawana', 'Tawanda', 'Tawanna', 'Tawny', 'Tawnya', 'Taya', 'Tayla',
    'Tayler', 'Taylor', 'Tea', 'Teagan', 'Teela', 'Teena', 'Tella',
    'Tempie', 'Tena', 'Tenika', 'Tenisha', 'Tennessee', 'Tennie',
    'Tennille', 'Tera', 'Teresa', 'Terese', 'Teressa', 'Teri', 'Terra',
    'Terri', 'Terrie', 'Terry', 'Tess', 'Tessa', 'Tessie', 'Texanna',
    'Texas', 'Texie', 'Thalia', 'Thea', 'Theda', 'Thekla', 'Thelma',
    'Theodocia', 'Theodora', 'Theodosia', 'Theola', 'Theresa', 'Therese',
    'Theresia', 'Theta', 'Thomasina', 'Thora', 'Thresa', 'Thursa', 'Thyra',
    'Tia', 'Tiana', 'Tianna', 'Tiara', 'Tiarra', 'Tiera', 'Tierra',
    'Tiesha', 'Tiffani', 'Tiffanie', 'Tiffany', 'Tilda', 'Tilla', 'Tillie',
    'Tina', 'Tiney', 'Tinie', 'Tinnie', 'Tiny', 'Tisa', 'Tisha', 'Tishie',
    'Tobi', 'Toby', 'Toccara', 'Tomasa', 'Tomeka', 'Tomika', 'Tommie',
    'Tonda', 'Toni', 'Tonia', 'Tonja', 'Tonya', 'Tori', 'Torie', 'Torrie',
    'Tory', 'Tosha', 'Toshiko', 'Towanda', 'Toya', 'Tracee', 'Tracey',
    'Traci', 'Tracie', 'Tracy', 'Treasure', 'Treena', 'Trena', 'Tresa',
    'Tressa', 'Tressie', 'Treva', 'Tricia', 'Trilby', 'Trina', 'Trinidad',
    'Trinity', 'Trish', 'Trisha', 'Trista', 'Tristan', 'Tristen', 'Trudi',
    'Trudie', 'Trudy', 'Trula', 'Tula', 'Twila', 'Twyla', 'Tyesha', 'Tyra',
    'Ula', 'Una', 'Unique', 'Unknown', 'Ura', 'Ursula', 'Vada', 'Val',
    'Valarie', 'Valencia', 'Valentina', 'Valentine', 'Valeria', 'Valerie',
    'Valery', 'Valinda', 'Vallie', 'Valorie', 'Vanesa', 'Vanessa', 'Vannie',
    'Vara', 'Vashti', 'Vassie', 'Veda', 'Vela', 'Velda', 'Velia', 'Vella',
    'Velma', 'Velva', 'Velvet', 'Vena', 'Venessa', 'Venice', 'Venie',
    'Venita', 'Vennie', 'Venus', 'Veola', 'Vera', 'Verda', 'Verdell',
    'Verdie', 'Verena', 'Vergie', 'Verla', 'Verlene', 'Verlie', 'Verna',
    'Verne', 'Vernell', 'Vernelle', 'Vernetta', 'Vernia', 'Vernice',
    'Vernie', 'Vernita', 'Verona', 'Veronica', 'Versa', 'Versie', 'Vertie',
    'Vessie', 'Vesta', 'Veta', 'Veva', 'Vicie', 'Vickey', 'Vicki', 'Vickie',
    'Vicky', 'Victoria', 'Victorine', 'Victory', 'Vicy', 'Vida', 'Vikki',
    'Villa', 'Vilma', 'Vina', 'Vincenza', 'Viney', 'Vinie', 'Vinnie',
    'Viola', 'Violet', 'Violeta', 'Violetta', 'Violette', 'Vira', 'Virdie',
    'Virgia', 'Virgie', 'Virginia', 'Viridiana', 'Vita', 'Viva', 'Vivian',
    'Viviana', 'Vivien', 'Vivienne', 'Vlasta', 'Vonda', 'Vonetta', 'Vonnie',
    'Wanda', 'Waneta', 'Wanita', 'Wava', 'Wende', 'Wendi', 'Wendy',
    'Whitley', 'Whitney', 'Wilda', 'Wilhelmina', 'Wilhelmine', 'Willa',
    'Willene', 'Willia', 'Willie', 'Williemae', 'Willodean', 'Willow',
    'Wilma', 'Windy', 'Winifred', 'Winnie', 'Winnifred', 'Winona', 'Winter',
    'Wynona', 'Xena', 'Ximena', 'Xiomara', 'Yadira', 'Yahaira', 'Yajaira',
    'Yamilet', 'Yamilex', 'Yareli', 'Yaretzi', 'Yaritza', 'Yasmeen',
    'Yasmin', 'Yasmine', 'Yazmin', 'Yesenia', 'Yessenia', 'Yetta',
    'Yolanda', 'Yolonda', 'Yoselin', 'Yoshiko', 'Yuliana', 'Yulisa',
    'Yulissa', 'Yuridia', 'Yvette', 'Yvonne', 'Zada', 'Zadie', 'Zaida',
    'Zana', 'Zandra', 'Zaniyah', 'Zara', 'Zaria', 'Zariah', 'Zela', 'Zelda',
    'Zelia', 'Zella', 'Zelma', 'Zelpha', 'Zena', 'Zenobia', 'Zeta', 'Zetta',
    'Zettie', 'Zhane', 'Zillah', 'Zilpah', 'Zilpha', 'Zina', 'Zion', 'Zita',
    'Zoa', 'Zoe', 'Zoey', 'Zoie', 'Zola', 'Zona', 'Zora', 'Zula',
    'Aaden', 'Aarav', 'Aaron', 'Ab', 'Abb', 'Abbott', 'Abdiel', 'Abdul',
    'Abdullah', 'Abe', 'Abel', 'Abelardo', 'Abie', 'Abner', 'Abraham',
    'Abram', 'Ace', 'Acey', 'Acie', 'Acy', 'Adalberto', 'Adam', 'Adams',
    'Adan', 'Add', 'Adelard', 'Adelbert', 'Aden', 'Adin', 'Aditya', 'Adlai',
    'Admiral', 'Adolf', 'Adolfo', 'Adolph', 'Adolphus', 'Adonis', 'Adrain',
    'Adrian', 'Adriel', 'Adrien', 'Adron', 'Aedan', 'Agustin', 'Agustus',
    'Ah', 'Ahmad', 'Ahmed', 'Aidan', 'Aiden', 'Aidyn', 'Aime', 'Akeem',
    'Al', 'Alan', 'Alanzo', 'Albert', 'Alberto', 'Albertus', 'Albin',
    'Albion', 'Alby', 'Alcee', 'Alcide', 'Alden', 'Aldo', 'Alec', 'Aleck',
    'Alejandro', 'Alek', 'Alessandro', 'Alex', 'Alexande', 'Alexander',
    'Alexandre', 'Alexandro', 'Alexis', 'Alexzander', 'Alf', 'Alferd',
    'Alfie', 'Alfonse', 'Alfonso', 'Alfonzo', 'Alford', 'Alfred', 'Alfredo',
    'Alger', 'Algernon', 'Algie', 'Algot', 'Ali', 'Alijah', 'Allan',
    'Allen', 'Allyn', 'Almer', 'Almon', 'Almond', 'Almus', 'Alois',
    'Alonso', 'Alonza', 'Alonzo', 'Aloys', 'Aloysius', 'Alpheus', 'Alphons',
    'Alphonse', 'Alphonso', 'Alphonsus', 'Alston', 'Alto', 'Alton', 'Alva',
    'Alvah', 'Alvan', 'Alvaro', 'Alver', 'Alvia', 'Alvie', 'Alvin', 'Alvis',
    'Alvy', 'Alwin', 'Amado', 'Amare', 'Amari', 'Amarion', 'Amasa',
    'Ambers', 'Ambrose', 'Americo', 'Amerigo', 'Amil', 'Amin', 'Amir',
    'Amit', 'Ammon', 'Amon', 'Amos', 'Ananias', 'Anastacio', 'Anatole',
    'Ancel', 'Ancil', 'Anders', 'Anderson', 'Andon', 'Andra', 'Andrae',
    'Andre', 'Andreas', 'Andres', 'Andrew', 'Andy', 'Anfernee', 'Angel',
    'Angelo', 'Angus', 'Anibal', 'Ansel', 'Anson', 'Anthoney', 'Anthony',
    'Antione', 'Antoine', 'Anton', 'Antone', 'Antonio', 'Antony', 'Antwain',
    'Antwan', 'Antwon', 'Anwar', 'Arba', 'Arbie', 'Arch', 'Archer',
    'Archibald', 'Archie', 'Ardell', 'Arden', 'Ari', 'Aric', 'Arjun',
    'Arlan', 'Arland', 'Arlen', 'Arley', 'Arlie', 'Arlin', 'Arlington',
    'Arlis', 'Arlo', 'Arlyn', 'Arman', 'Armand', 'Armando', 'Armani',
    'Armin', 'Armond', 'Armstead', 'Arnav', 'Arne', 'Arnett', 'Arnie',
    'Arno', 'Arnold', 'Arnoldo', 'Arnulfo', 'Aron', 'Arron', 'Arsenio',
    'Art', 'Arther', 'Arthor', 'Arthur', 'Artie', 'Artis', 'Arturo',
    'Arvel', 'Arvid', 'Arvil', 'Arvin', 'Arvo', 'Aryan', 'Asa', 'Asberry',
    'Asbury', 'Ashby', 'Asher', 'Ashton', 'Atha', 'Atlas', 'Atticus',
    'Attilio', 'Aubra', 'Aubrey', 'Audie', 'Audley', 'Audy', 'August',
    'Auguste', 'Augustin', 'Augustine', 'Augustus', 'Aurelio', 'Aurthur',
    'Austen', 'Austin', 'Auston', 'Austyn', 'Auther', 'Author', 'Authur',
    'Autry', 'Avery', 'Avon', 'Axel', 'Ayaan', 'Aydan', 'Ayden', 'Aydin',
    'Babe', 'Babyboy', 'Bailey', 'Baker', 'Baldwin', 'Ballard', 'Banks',
    'Barnard', 'Barnett', 'Barney', 'Barnie', 'Baron', 'Barrett', 'Barrie',
    'Barron', 'Barry', 'Bart', 'Bartholomew', 'Bartley', 'Barton', 'Bascom',
    'Basil', 'Baxter', 'Bayard', 'Beau', 'Beckett', 'Beckham', 'Bedford',
    'Beecher', 'Bell', 'Belton', 'Ben', 'Benard', 'Benedict', 'Benito',
    'Benjaman', 'Benjamen', 'Benjamin', 'Benjamine', 'Benji', 'Benjiman',
    'Benjman', 'Bennett', 'Bennie', 'Benny', 'Benson', 'Bentley', 'Benton',
    'Berkley', 'Berlin', 'Bernard', 'Bernardo', 'Bernhard', 'Bernie',
    'Berry', 'Bert', 'Bertie', 'Berton', 'Bertram', 'Bertrand', 'Beryl',
    'Bethel', 'Bilal', 'Bill', 'Billie', 'Billy', 'Bird', 'Birt', 'Bishop',
    'Bjorn', 'Blain', 'Blaine', 'Blair', 'Blaise', 'Blake', 'Blanchard',
    'Blane', 'Blas', 'Blaze', 'Bliss', 'Bluford', 'Bo', 'Bob', 'Bobbie',
    'Bobby', 'Bode', 'Bolden', 'Booker', 'Boone', 'Boris', 'Bose', 'Boss',
    'Boston', 'Bowman', 'Boyce', 'Boyd', 'Boysie', 'Brad', 'Braden',
    'Bradford', 'Bradley', 'Bradly', 'Brady', 'Bradyn', 'Braeden',
    'Braedon', 'Braiden', 'Brain', 'Branch', 'Brandan', 'Branden',
    'Brandin', 'Brandon', 'Brandt', 'Brandy', 'Brandyn', 'Brannon',
    'Branson', 'Brant', 'Brantley', 'Braulio', 'Braxton', 'Brayan',
    'Brayden', 'Braydon', 'Braylen', 'Braylon', 'Brendan', 'Brenden',
    'Brendon', 'Brennan', 'Brennen', 'Brennon', 'Brent', 'Brenton', 'Bret',
    'Brett', 'Brian', 'Brice', 'Bridger', 'Brien', 'Brion', 'Britt',
    'Brittany', 'Britton', 'Brock', 'Broderick', 'Brodie', 'Brody',
    'Brogan', 'Bronson', 'Brook', 'Brooks', 'Brown', 'Bruce', 'Bruno',
    'Bryan', 'Bryant', 'Bryce', 'Brycen', 'Bryon', 'Bryson', 'Bryton',
    'Buck', 'Bud', 'Budd', 'Buddie', 'Buddy', 'Buel', 'Buell', 'Buford',
    'Bunk', 'Burdette', 'Buren', 'Burgess', 'Burk', 'Burke', 'Burl',
    'Burleigh', 'Burley', 'Burnell', 'Burnett', 'Burney', 'Burnice',
    'Burnie', 'Burns', 'Burr', 'Burrel', 'Burrell', 'Burt', 'Burton',
    'Bush', 'Buster', 'Butch', 'Butler', 'Bynum', 'Byrd', 'Byron', 'Cade',
    'Caden', 'Cael', 'Caesar', 'Caiden', 'Cain', 'Cal', 'Cale', 'Caleb',
    'Calhoun', 'Callie', 'Callum', 'Calvin', 'Cam', 'Camden', 'Cameron',
    'Camilo', 'Campbell', 'Camren', 'Camron', 'Camryn', 'Candido', 'Cannon',
    'Canyon', 'Cap', 'Captain', 'Carey', 'Carl', 'Carleton', 'Carlie',
    'Carlisle', 'Carlo', 'Carlos', 'Carlton', 'Carlyle', 'Carmel',
    'Carmelo', 'Carmen', 'Carmine', 'Carnell', 'Carrie', 'Carrol',
    'Carroll', 'Carsen', 'Carson', 'Carter', 'Cary', 'Cas', 'Case', 'Casen',
    'Casey', 'Cash', 'Casimer', 'Casimir', 'Casimiro', 'Cason', 'Casper',
    'Cass', 'Cassidy', 'Cassie', 'Cassius', 'Caswell', 'Cato', 'Cayden',
    'Ceasar', 'Cecil', 'Cedric', 'Cedrick', 'Celestino', 'Cephus', 'Cesar',
    'Ceylon', 'Chace', 'Chad', 'Chadd', 'Chadrick', 'Chadwick', 'Chaim',
    'Chalmer', 'Chalmers', 'Champ', 'Chance', 'Chancey', 'Chancy',
    'Chandler', 'Channing', 'Charle', 'Charles', 'Charley', 'Charlie',
    'Charls', 'Charlton', 'Charly', 'Chas', 'Chase', 'Chauncey', 'Chauncy',
    'Chaz', 'Che', 'Chesley', 'Chester', 'Chet', 'Cheyenne', 'Chin', 'Chip',
    'Chris', 'Christ', 'Christian', 'Christina', 'Christion', 'Christop',
    'Christoper', 'Christophe', 'Christopher', 'Chuck', 'Cicero', 'Clabe',
    'Claiborne', 'Clair', 'Clarance', 'Clare', 'Clarence', 'Clark',
    'Clarke', 'Clarnce', 'Claud', 'Claude', 'Claudie', 'Claudio',
    'Claudius', 'Claus', 'Clay', 'Clayton', 'Clearence', 'Cleave', 'Clell',
    'Clem', 'Clemence', 'Clemens', 'Clement', 'Clemente', 'Clemmie',
    'Clemon', 'Cleo', 'Cleon', 'Cletus', 'Cleve', 'Cleveland', 'Clide',
    'Cliff', 'Clifford', 'Clifton', 'Clint', 'Clinton', 'Clive', 'Clovis',
    'Cloyd', 'Clyde', 'Coby', 'Codey', 'Codi', 'Codie', 'Cody', 'Coen',
    'Cohen', 'Colbert', 'Colby', 'Cole', 'Coleman', 'Coleton', 'Coley',
    'Colie', 'Colin', 'Collie', 'Collier', 'Collin', 'Collins', 'Collis',
    'Colon', 'Colonel', 'Colt', 'Colten', 'Colter', 'Colton', 'Columbus',
    'Colvin', 'Commodore', 'Con', 'Conard', 'Conley', 'Conner', 'Connie',
    'Connor', 'Conor', 'Conrad', 'Constantine', 'Conway', 'Coolidge',
    'Cooper', 'Corbett', 'Corbin', 'Cordaro', 'Cordell', 'Cordero', 'Corey',
    'Cornel', 'Cornelious', 'Cornelius', 'Cornell', 'Corry', 'Cortez',
    'Cortney', 'Corwin', 'Cory', 'Cosmo', 'Coty', 'Council', 'Courtland',
    'Courtney', 'Coy', 'Craig', 'Crawford', 'Creed', 'Cris', 'Cristian',
    'Cristobal', 'Cristofer', 'Cristopher', 'Crockett', 'Cruz', 'Cullen',
    'Curley', 'Curt', 'Curtis', 'Curtiss', 'Cyril', 'Cyrus', 'Dabney',
    'Dakoda', 'Dakota', 'Dakotah', 'Dale', 'Dallas', 'Dallin', 'Dalton',
    'Dalvin', 'Damarcus', 'Damari', 'Damarion', 'Dameon', 'Damian',
    'Damien', 'Damion', 'Damon', 'Damond', 'Dan', 'Dana', 'Dandre', 'Dane',
    'Dangelo', 'Danial', 'Daniel', 'Dann', 'Dannie', 'Danniel', 'Danny',
    'Dante', 'Daquan', 'Darby', 'Darcy', 'Darell', 'Daren', 'Darian',
    'Darien', 'Darin', 'Dario', 'Darion', 'Darius', 'Darl', 'Darnell',
    'Darold', 'Daron', 'Darrel', 'Darrell', 'Darren', 'Darrian', 'Darrick',
    'Darrien', 'Darrin', 'Darrion', 'Darrius', 'Darron', 'Darry', 'Darryl',
    'Darryle', 'Darryll', 'Darryn', 'Darvin', 'Darwin', 'Darwyn', 'Daryl',
    'Daryle', 'Daryn', 'Dashawn', 'Daulton', 'Daunte', 'Davante', 'Dave',
    'Davey', 'Davian', 'David', 'Davie', 'Davin', 'Davion', 'Davis',
    'Davon', 'Davonta', 'Davonte', 'Davy', 'Dawson', 'Dax', 'Daxton',
    'Dayne', 'Dayton', 'Deacon', 'Dean', 'Deandre', 'Deane', 'Deangelo',
    'Deante', 'Declan', 'Dedric', 'Dedrick', 'Deegan', 'Deforest', 'Deion',
    'Dejon', 'Dejuan', 'Del', 'Delano', 'Delbert', 'Dell', 'Della', 'Delma',
    'Delmar', 'Delmas', 'Delmer', 'Delmus', 'Delos', 'Delphin', 'Delton',
    'Delvin', 'Delwin', 'Demarco', 'Demarcus', 'Demario', 'Demarion',
    'Demetri', 'Demetric', 'Demetrios', 'Demetrius', 'Demian', 'Demond',
    'Demonte', 'Dempsey', 'Denis', 'Dennie', 'Dennis', 'Denny', 'Denton',
    'Denver', 'Denzel', 'Denzell', 'Denzil', 'Deon', 'Deondre', 'Deonta',
    'Deontae', 'Deonte', 'Dequan', 'Derald', 'Dereck', 'Derek', 'Dereon',
    'Deric', 'Derick', 'Derik', 'Derl', 'Deron', 'Derrek', 'Derrell',
    'Derrick', 'Derwin', 'Deryl', 'Desean', 'Deshaun', 'Deshawn', 'Desi',
    'Desmond', 'Dessie', 'Destin', 'Destry', 'Devan', 'Devante', 'Devaughn',
    'Deven', 'Devin', 'Devon', 'Devonta', 'Devontae', 'Devonte', 'Devyn',
    'Deward', 'Dewayne', 'Dewey', 'Dewitt', 'Dexter', 'Diallo', 'Diamond',
    'Diane', 'Dickie', 'Diego', 'Dijon', 'Dilan', 'Dillan', 'Dillard',
    'Dillion', 'Dillon', 'Dimitri', 'Dimitrios', 'Dink', 'Dino', 'Dion',
    'Dionicio', 'Dionte', 'Dirk', 'Dixon', 'Doc', 'Dock', 'Doctor', 'Doll',
    'Dolph', 'Dolphus', 'Domenic', 'Domenick', 'Domenico', 'Domingo',
    'Dominic', 'Dominick', 'Dominik', 'Don', 'Donaciano', 'Donal', 'Donald',
    'Donat', 'Donato', 'Donavan', 'Donavon', 'Dondre', 'Donell', 'Donn',
    'Donnell', 'Donnie', 'Donny', 'Donovan', 'Donta', 'Dontae', 'Donte',
    'Dora', 'Dorian', 'Dorman', 'Dorr', 'Dorris', 'Dorsey', 'Doss', 'Doug',
    'Douglas', 'Douglass', 'Dow', 'Doyle', 'Dozier', 'Drake', 'Draven',
    'Drew', 'Drury', 'Duane', 'Duard', 'Dudley', 'Duff', 'Duke', 'Duncan',
    'Durell', 'Durrell', 'Durward', 'Durwood', 'Dustan', 'Dustin', 'Dusty',
    'Dustyn', 'Duwayne', 'Dwain', 'Dwaine', 'Dwane', 'Dwayne', 'Dwight',
    'Dwyane', 'Dylan', 'Dyllan', 'Dylon', 'Ean', 'Earl', 'Earle', 'Earley',
    'Earlie', 'Early', 'Earnest', 'Easton', 'Ebb', 'Ebbie', 'Eben',
    'Ebenezer', 'Eber', 'Ebert', 'Ed', 'Edd', 'Eddie', 'Eddy', 'Eden',
    'Edgar', 'Edgardo', 'Edie', 'Edison', 'Edmon', 'Edmond', 'Edmund',
    'Edsel', 'Edson', 'Eduardo', 'Edw', 'Edward', 'Edwardo', 'Edwin',
    'Effie', 'Efrain', 'Efrem', 'Efren', 'Egbert', 'Einar', 'Eino', 'Elam',
    'Elbert', 'Elbridge', 'Elby', 'Elden', 'Elder', 'Eldon', 'Eldred',
    'Eldridge', 'Elex', 'Elgie', 'Elgin', 'Eli', 'Elian', 'Elias', 'Elick',
    'Elie', 'Eliezer', 'Eliga', 'Eligah', 'Elige', 'Elihu', 'Elijah',
    'Eliot', 'Eliseo', 'Elisha', 'Elizah', 'Ell', 'Ellery', 'Elliot',
    'Elliott', 'Ellis', 'Ellison', 'Ellsworth', 'Ellwood', 'Elmer', 'Elmo',
    'Elmore', 'Elon', 'Elonzo', 'Eloy', 'Elroy', 'Elsworth', 'Elton',
    'Elvin', 'Elvis', 'Elwin', 'Elwood', 'Elwyn', 'Ely', 'Elza', 'Elzie',
    'Elzy', 'Emanuel', 'Emerson', 'Emery', 'Emett', 'Emil', 'Emile',
    'Emiliano', 'Emilio', 'Emit', 'Emma', 'Emmanuel', 'Emmet', 'Emmett',
    'Emmit', 'Emmitt', 'Emmons', 'Emory', 'Emry', 'Encarnacion', 'Ennis',
    'Enoch', 'Enos', 'Enrico', 'Enrique', 'Enzo', 'Ephraim', 'Ephram',
    'Ephriam', 'Epifanio', 'Erasmo', 'Erasmus', 'Erastus', 'Erby', 'Eric',
    'Erich', 'Erick', 'Erie', 'Erik', 'Erin', 'Erland', 'Erle', 'Erling',
    'Ernest', 'Ernesto', 'Ernie', 'Ernst', 'Errol', 'Ervin', 'Erving',
    'Erwin', 'Esau', 'Esco', 'Esequiel', 'Esker', 'Esley', 'Essex',
    'Esteban', 'Estel', 'Estes', 'Estevan', 'Estill', 'Eston', 'Ethan',
    'Ethelbert', 'Ethen', 'Eugene', 'Eugenio', 'Eusebio', 'Eustace', 'Evan',
    'Evander', 'Evans', 'Evelyn', 'Everet', 'Everett', 'Everette', 'Evert',
    'Evertt', 'Ewald', 'Ewart', 'Ewell', 'Ewin', 'Ewing', 'Ezekiel',
    'Ezell', 'Ezequiel', 'Ezra', 'Ezzard', 'Fabian', 'Faron', 'Farrell',
    'Farris', 'Fate', 'Faustino', 'Fayette', 'Fed', 'Federico', 'Felipe',
    'Felix', 'Felton', 'Fenton', 'Ferd', 'Ferdinand', 'Ferman', 'Fernand',
    'Fernando', 'Ferrell', 'Ferris', 'Festus', 'Fidel', 'Fidencio',
    'Fielding', 'Finis', 'Finley', 'Finn', 'Finnegan', 'Firman', 'Fisher',
    'Fitzgerald', 'Fitzhugh', 'Fleet', 'Flem', 'Fleming', 'Fletcher',
    'Flint', 'Florencio', 'Florentino', 'Florian', 'Floy', 'Floyd', 'Foch',
    'Ford', 'Forest', 'Forrest', 'Foster', 'Fount', 'Foy', 'Frances',
    'Francesco', 'Francis', 'Francisco', 'Franco', 'Frank', 'Frankie',
    'Franklin', 'Franklyn', 'Franz', 'Frazier', 'Fred', 'Freddie', 'Freddy',
    'Frederic', 'Frederick', 'Fredie', 'Fredric', 'Fredrick', 'Fredy',
    'Freeman', 'Fremont', 'French', 'Friend', 'Fritz', 'Fuller', 'Fulton',
    'Furman', 'Gabe', 'Gabriel', 'Gael', 'Gaetano', 'Gage', 'Gaige', 'Gail',
    'Gaines', 'Gaither', 'Gale', 'Galen', 'Gannon', 'Gardner', 'Garett',
    'Garey', 'Garfield', 'Garland', 'Garner', 'Garnet', 'Garnett', 'Garold',
    'Garret', 'Garrett', 'Garrick', 'Garrison', 'Garry', 'Garth', 'Garvin',
    'Gary', 'Gasper', 'Gaston', 'Gauge', 'Gaven', 'Gavin', 'Gavyn', 'Gay',
    'Gayle', 'Gaylen', 'Gaylon', 'Gaylord', 'Gearld', 'Geary', 'Gee',
    'Genaro', 'Gene', 'General', 'Genie', 'Gennaro', 'Geno', 'Geo', 'Geoff',
    'Geoffrey', 'George', 'Georgie', 'Geovanni', 'Gerald', 'Geraldo',
    'Gerard', 'Gerardo', 'Gerhard', 'Gerhardt', 'Germaine', 'German',
    'Gerold', 'Gerrit', 'Gerry', 'Giancarlo', 'Gianni', 'Gibson', 'Gideon',
    'Gifford', 'Gil', 'Gilbert', 'Gilberto', 'Giles', 'Gilford', 'Gilman',
    'Gilmer', 'Gilmore', 'Gino', 'Giovani', 'Giovanni', 'Giovanny',
    'Giuseppe', 'Gladstone', 'Glen', 'Glendon', 'Glenn', 'Glenwood',
    'Glover', 'Glynn', 'Godfrey', 'Goebel', 'Golden', 'Gonzalo', 'Gorden',
    'Gordon', 'Gorge', 'Gottlieb', 'Governor', 'Grady', 'Grafton', 'Graham',
    'Grant', 'Granville', 'Graves', 'Gray', 'Graydon', 'Grayling',
    'Grayson', 'Green', 'Greene', 'Greg', 'Gregg', 'Greggory', 'Gregorio',
    'Gregory', 'Greyson', 'Griffin', 'Griffith', 'Grove', 'Grover', 'Guido',
    'Guilford', 'Guillermo', 'Gunnar', 'Gunner', 'Gurney', 'Gus', 'Guss',
    'Gussie', 'Gust', 'Gustaf', 'Gustav', 'Gustave', 'Gustavo', 'Gustavus',
    'Guthrie', 'Guy', 'Haden', 'Hadley', 'Haiden', 'Hakeem', 'Hakim', 'Hal',
    'Halbert', 'Hale', 'Hall', 'Halley', 'Hallie', 'Halsey', 'Ham',
    'Hamilton', 'Hamp', 'Hampton', 'Hamza', 'Handy', 'Hank', 'Hans',
    'Hansel', 'Hansford', 'Hanson', 'Harden', 'Hardie', 'Hardin', 'Harding',
    'Hardy', 'Harl', 'Harlan', 'Harland', 'Harlen', 'Harley', 'Harlie',
    'Harlon', 'Harlow', 'Harm', 'Harman', 'Harmon', 'Harold', 'Harper',
    'Harrell', 'Harrie', 'Harris', 'Harrison', 'Harrold', 'Harry', 'Hart',
    'Hartley', 'Hartwell', 'Harve', 'Harvey', 'Harvie', 'Harvy', 'Hasan',
    'Haskell', 'Hassan', 'Hattie', 'Haven', 'Hayden', 'Hayes', 'Hays',
    'Hayward', 'Haywood', 'Hazen', 'Heath', 'Heber', 'Hebert', 'Hector',
    'Helmer', 'Hence', 'Henderson', 'Henery', 'Henri', 'Henry', 'Herb',
    'Herbert', 'Heriberto', 'Herman', 'Hermann', 'Hermon', 'Hernan',
    'Herschel', 'Hershel', 'Hershell', 'Hervey', 'Heyward', 'Hezekiah',
    'Hezzie', 'Hideo', 'Hilario', 'Hilary', 'Hilbert', 'Hill', 'Hillard',
    'Hillary', 'Hillery', 'Hilliard', 'Hilmer', 'Hilton', 'Hiram',
    'Hiroshi', 'Hjalmar', 'Hjalmer', 'Hobart', 'Hobert', 'Hobson', 'Hoke',
    'Holden', 'Holland', 'Hollie', 'Hollis', 'Holmes', 'Homer', 'Hoover',
    'Hope', 'Horace', 'Horacio', 'Horatio', 'Horton', 'Hosea', 'Hosie',
    'Hosteen', 'Houston', 'Howard', 'Howell', 'Hoy', 'Hoyt', 'Hubbard',
    'Hubert', 'Hudson', 'Huey', 'Hugh', 'Hughes', 'Hughey', 'Hughie',
    'Hugo', 'Humberto', 'Humphrey', 'Hung', 'Hunt', 'Hunter', 'Hurbert',
    'Hurley', 'Huston', 'Huy', 'Hyman', 'Hymen', 'Hyrum', 'Ian', 'Ibrahim',
    'Ida', 'Ignacio', 'Ignatius', 'Ignatz', 'Ike', 'Illya', 'Imanol',
    'Immanuel', 'Infant', 'Ingram', 'Ira', 'Irl', 'Irven', 'Irvin',
    'Irvine', 'Irving', 'Irwin', 'Isaac', 'Isaak', 'Isadore', 'Isai',
    'Isaiah', 'Isaias', 'Isam', 'Ishaan', 'Isham', 'Ishmael', 'Isiah',
    'Isidor', 'Isidore', 'Isidro', 'Ismael', 'Isom', 'Israel', 'Isreal',
    'Issac', 'Iva', 'Ivan', 'Iver', 'Iverson', 'Ivey', 'Ivor', 'Ivory',
    'Ivy', 'Izaiah', 'Izayah', 'Jabari', 'Jabbar', 'Jabez', 'Jace', 'Jack',
    'Jackson', 'Jacky', 'Jacob', 'Jacoby', 'Jacques', 'Jacquez', 'Jade',
    'Jaden', 'Jadiel', 'Jadon', 'Jadyn', 'Jaeden', 'Jagger', 'Jaheem',
    'Jaheim', 'Jahiem', 'Jahir', 'Jaiden', 'Jaidyn', 'Jaime', 'Jaimie',
    'Jair', 'Jairo', 'Jajuan', 'Jake', 'Jakob', 'Jakobe', 'Jaleel', 'Jalen',
    'Jalon', 'Jamaal', 'Jamal', 'Jamar', 'Jamarcus', 'Jamari', 'Jamarion',
    'Jame', 'Jameel', 'Jamel', 'James', 'Jameson', 'Jamey', 'Jamie',
    'Jamil', 'Jamin', 'Jamir', 'Jamison', 'Jammie', 'Jan', 'Jaquan',
    'Jaquez', 'Jarad', 'Jared', 'Jaren', 'Jaret', 'Jarett', 'Jarod',
    'Jaron', 'Jarrad', 'Jarred', 'Jarrell', 'Jarret', 'Jarrett', 'Jarrod',
    'Jarvis', 'Jase', 'Jasen', 'Jasiah', 'Jason', 'Jasper', 'Javen',
    'Javier', 'Javion', 'Javon', 'Javonte', 'Jax', 'Jaxen', 'Jaxon',
    'Jaxson', 'Jaxton', 'Jay', 'Jayce', 'Jaycob', 'Jaydan', 'Jayden',
    'Jaydin', 'Jaydon', 'Jaylan', 'Jaylen', 'Jaylin', 'Jaylon', 'Jayme',
    'Jaymes', 'Jayson', 'Jayvion', 'Jayvon', 'Jean', 'Jeb', 'Jed',
    'Jedediah', 'Jedidiah', 'Jeff', 'Jefferey', 'Jefferson', 'Jeffery',
    'Jeffie', 'Jeffrey', 'Jeffry', 'Jelani', 'Jemal', 'Jennings', 'Jens',
    'Jensen', 'Jep', 'Jeptha', 'Jerad', 'Jerald', 'Jeramiah', 'Jeramie',
    'Jeramy', 'Jere', 'Jered', 'Jerel', 'Jereme', 'Jeremey', 'Jeremiah',
    'Jeremie', 'Jeremy', 'Jerimiah', 'Jerimy', 'Jermain', 'Jermaine',
    'Jermey', 'Jerod', 'Jerold', 'Jerome', 'Jeromy', 'Jerrad', 'Jerrel',
    'Jerrell', 'Jerrod', 'Jerrold', 'Jerry', 'Jess', 'Jesse', 'Jessee',
    'Jessie', 'Jessy', 'Jesus', 'Jethro', 'Jett', 'Jettie', 'Jevon',
    'Jewell', 'Jiles', 'Jim', 'Jimmie', 'Jimmy', 'Joaquin', 'Job', 'Jobe',
    'Joe', 'Joel', 'Joeseph', 'Joesph', 'Joey', 'Johan', 'Johathan', 'John',
    'Johnathan', 'Johnathon', 'Johney', 'Johnie', 'Johnnie', 'Johnny',
    'Johnpaul', 'Johnson', 'Johny', 'Jon', 'Jonah', 'Jonas', 'Jonatan',
    'Jonathan', 'Jonathon', 'Jones', 'Jonnie', 'Jordan', 'Jorden', 'Jordi',
    'Jordon', 'Jordy', 'Jordyn', 'Jorge', 'Jory', 'Jose', 'Josef',
    'Joseluis', 'Joseph', 'Josephus', 'Josh', 'Joshua', 'Joshuah', 'Josiah',
    'Josue', 'Jovan', 'Jovani', 'Jovanni', 'Jovanny', 'Jovany', 'Joy',
    'Juan', 'Judah', 'Judd', 'Jude', 'Judge', 'Judson', 'Juelz', 'Jule',
    'Jules', 'Julian', 'Julien', 'Julio', 'Julious', 'Julius', 'Juluis',
    'Junior', 'Junious', 'Junius', 'Justen', 'Justice', 'Justin', 'Juston',
    'Justus', 'Justyn', 'Juwan', 'Kade', 'Kadeem', 'Kaden', 'Kadin',
    'Kadyn', 'Kaeden', 'Kael', 'Kahlil', 'Kai', 'Kaiden', 'Kale', 'Kaleb',
    'Kalel', 'Kalen', 'Kalvin', 'Kamari', 'Kamden', 'Kameron', 'Kamren',
    'Kamron', 'Kamryn', 'Kane', 'Kanye', 'Kareem', 'Kareen', 'Karim',
    'Karl', 'Karson', 'Karter', 'Kasen', 'Kasey', 'Kash', 'Kason', 'Kavon',
    'Kayden', 'Kaye', 'Kayson', 'Kazuo', 'Keagan', 'Keandre', 'Keanu',
    'Keaton', 'Keegan', 'Keenan', 'Keenen', 'Kegan', 'Keifer', 'Keion',
    'Keith', 'Kelan', 'Kelby', 'Kellan', 'Kellen', 'Kelley', 'Kelly',
    'Kelsey', 'Kelton', 'Kelvin', 'Kem', 'Ken', 'Kenan', 'Kendal',
    'Kendall', 'Kendell', 'Kendrick', 'Kenji', 'Kennard', 'Kennedy',
    'Kenneth', 'Kenney', 'Kennith', 'Kennth', 'Kenny', 'Kent', 'Kenton',
    'Kenya', 'Kenyatta', 'Kenyon', 'Keon', 'Kermit', 'Kerry', 'Kerwin',
    'Keshaun', 'Keshawn', 'Kevan', 'Keven', 'Kevin', 'Kevon', 'Keyon',
    'Keyshawn', 'Khalid', 'Khalil', 'Khari', 'Khiry', 'Kian', 'Kiara',
    'Kiefer', 'Kiel', 'Kieran', 'Kieth', 'Kiley', 'Killian', 'Kim',
    'Kimball', 'Kimberly', 'King', 'Kingston', 'Kinte', 'Kip', 'Kipp',
    'Kirby', 'Kirk', 'Kirt', 'Kit', 'Kiyoshi', 'Knox', 'Knute', 'Kobe',
    'Koby', 'Koda', 'Kody', 'Koen', 'Kolby', 'Kole', 'Kolten', 'Kolton',
    'Konner', 'Konnor', 'Korbin', 'Kordell', 'Korey', 'Kory', 'Kraig',
    'Kris', 'Krish', 'Kristen', 'Kristian', 'Kristin', 'Kristofer',
    'Kristoffer', 'Kristopher', 'Kunta', 'Kurt', 'Kurtis', 'Kwame', 'Kyan',
    'Kylan', 'Kyle', 'Kyler', 'Kymani', 'Kyree', 'Kyson', 'Lacey', 'Lacy',
    'Ladarius', 'Laddie', 'Lafayette', 'Lafe', 'Lamar', 'Lamarcus',
    'Lambert', 'Lamont', 'Lamonte', 'Lance', 'Landan', 'Landen', 'Landin',
    'Landon', 'Landyn', 'Lane', 'Lannie', 'Lanny', 'Laquan', 'Lark',
    'Larkin', 'Laron', 'Larry', 'Lars', 'Larue', 'Lary', 'Lashawn',
    'Latrell', 'Laurance', 'Laurel', 'Laurence', 'Lavar', 'Lavern',
    'Laverne', 'Lavon', 'Lawerence', 'Lawrance', 'Lawrence', 'Lawson',
    'Lawton', 'Lawyer', 'Layne', 'Layton', 'Lazaro', 'Le', 'Lea', 'Leamon',
    'Leander', 'Leandro', 'Lee', 'Leeroy', 'Leif', 'Leigh', 'Leighton',
    'Leland', 'Lem', 'Lemmie', 'Lemon', 'Lemuel', 'Len', 'Lena', 'Lenard',
    'Lennie', 'Lennon', 'Lenny', 'Lenon', 'Lenord', 'Lenwood', 'Leo',
    'Leon', 'Leonard', 'Leonardo', 'Leonce', 'Leonel', 'Leonidas',
    'Leopold', 'Leopoldo', 'Leroy', 'Les', 'Lesley', 'Leslie', 'Less',
    'Lessie', 'Lester', 'Levar', 'Levern', 'Levi', 'Levie', 'Levin',
    'Levon', 'Levy', 'Lew', 'Lewis', 'Lex', 'Lexie', 'Liam', 'Lige',
    'Lilburn', 'Lillard', 'Lim', 'Lincoln', 'Lindbergh', 'Lindell',
    'Linden', 'Lindsay', 'Lindsey', 'Lindy', 'Link', 'Linn', 'Linnie',
    'Linton', 'Linus', 'Linwood', 'Linzy', 'Lionel', 'Lisandro', 'Lish',
    'Lisle', 'Liston', 'Little', 'Littleton', 'Llewellyn', 'Lloyd', 'Logan',
    'Lon', 'London', 'Lone', 'Loney', 'Long', 'Lonie', 'Lonnie', 'Lonny',
    'Lonzo', 'Lora', 'Loran', 'Loren', 'Lorenz', 'Lorenza', 'Lorenzo',
    'Lorin', 'Loring', 'Lorne', 'Lott', 'Lou', 'Louie', 'Louis', 'Love',
    'Lovell', 'Lovett', 'Lovie', 'Lowell', 'Loy', 'Loyal', 'Loyd', 'Luc',
    'Luca', 'Lucas', 'Lucian', 'Luciano', 'Lucien', 'Lucio', 'Lucious',
    'Lucius', 'Lucky', 'Ludwig', 'Lue', 'Luigi', 'Luis', 'Luka', 'Lukas',
    'Luke', 'Lula', 'Lum', 'Lupe', 'Luster', 'Lute', 'Luther', 'Luverne',
    'Lydell', 'Lyle', 'Lyman', 'Lyn', 'Lyndon', 'Lynn', 'Lynwood', 'Lyric',
    'Mac', 'Macarthur', 'Mace', 'Maceo', 'Mack', 'Mackenzie', 'Madden',
    'Maddox', 'Maddux', 'Madison', 'Mae', 'Mahlon', 'Major', 'Makai',
    'Makhi', 'Mal', 'Malachi', 'Malakai', 'Malaki', 'Malcolm', 'Malcom',
    'Male', 'Malik', 'Malvin', 'Mamie', 'Manford', 'Manley', 'Manly',
    'Mannie', 'Manning', 'Mansfield', 'Manson', 'Manuel', 'Marc', 'Marcel',
    'Marcelino', 'Marcell', 'Marcello', 'Marcellus', 'Marcelo', 'Marchello',
    'Marco', 'Marcos', 'Marcus', 'Margarito', 'Mariano', 'Mario', 'Marion',
    'Marius', 'Mark', 'Markel', 'Markell', 'Markus', 'Marland', 'Marley',
    'Marlin', 'Marlo', 'Marlon', 'Marlyn', 'Marques', 'Marquez', 'Marquis',
    'Marquise', 'Marrion', 'Marsh', 'Marshal', 'Marshall', 'Mart',
    'Martell', 'Martez', 'Martin', 'Marty', 'Marvin', 'Masao', 'Mason',
    'Mat', 'Mateo', 'Math', 'Mathew', 'Mathews', 'Mathias', 'Matias',
    'Matt', 'Matteo', 'Matthew', 'Matthias', 'Maurice', 'Mauricio', 'Mauro',
    'Maury', 'Maverick', 'Max', 'Maxie', 'Maxim', 'Maximilian',
    'Maximiliano', 'Maximillian', 'Maximo', 'Maximus', 'Maxwell', 'Maxx',
    'May', 'Maynard', 'Mayo', 'Mcarthur', 'Mckinley', 'Mearl', 'Mekhi',
    'Mel', 'Melbourne', 'Mell', 'Melton', 'Melville', 'Melvin', 'Melvyn',
    'Memphis', 'Menachem', 'Mercer', 'Merl', 'Merle', 'Merlin', 'Merlyn',
    'Merrill', 'Merritt', 'Merton', 'Mervin', 'Mervyn', 'Merwin', 'Messiah',
    'Metro', 'Meyer', 'Micah', 'Michael', 'Michal', 'Michale', 'Micheal',
    'Michel', 'Michial', 'Mickey', 'Micky', 'Miguel', 'Miguelangel',
    'Mikal', 'Mike', 'Mikeal', 'Mikel', 'Mikhail', 'Milan', 'Milas',
    'Milburn', 'Miles', 'Milford', 'Millard', 'Miller', 'Mills', 'Milo',
    'Milton', 'Miner', 'Minor', 'Minoru', 'Misael', 'Mitch', 'Mitchel',
    'Mitchell', 'Moe', 'Mohamed', 'Mohammad', 'Mohammed', 'Moises',
    'Monroe', 'Mont', 'Montana', 'Monte', 'Montel', 'Montgomery', 'Montie',
    'Montrell', 'Monty', 'Moody', 'Mordechai', 'Morgan', 'Morris',
    'Mortimer', 'Morton', 'Mose', 'Moses', 'Moshe', 'Muhammad', 'Murdock',
    'Murl', 'Murphy', 'Murray', 'Murry', 'Mustafa', 'Mychal', 'Myer',
    'Mykel', 'Myles', 'Myrl', 'Myron', 'Myrtle', 'Najee', 'Nakia', 'Namon',
    'Napoleon', 'Nash', 'Nasir', 'Nat', 'Nathan', 'Nathanael', 'Nathanial',
    'Nathaniel', 'Nathen', 'Neal', 'Ned', 'Needham', 'Neely', 'Nehemiah',
    'Neil', 'Nello', 'Nels', 'Nelson', 'Nery', 'Nestor', 'Nevin', 'Newell',
    'Newman', 'Newt', 'Newton', 'Nicholas', 'Nicholaus', 'Nick', 'Nicklaus',
    'Nickolas', 'Nicky', 'Nico', 'Nicolas', 'Nigel', 'Nikhil', 'Nikko',
    'Niko', 'Nikolai', 'Nikolas', 'Nile', 'Niles', 'Nils', 'Nim', 'Noah',
    'Noble', 'Noe', 'Noel', 'Nolan', 'Nolen', 'Norbert', 'Norberto',
    'Norman', 'Normand', 'Norris', 'North', 'Norton', 'Norval', 'Norwood',
    'Nunzio', 'Oakley', 'Obe', 'Obed', 'Obie', 'Ocie', 'Octave', 'Octavio',
    'Octavius', 'Oda', 'Oddie', 'Odell', 'Odie', 'Odin', 'Odis', 'Odus',
    'Offie', 'Ogden', 'Okey', 'Ola', 'Olaf', 'Olan', 'Oland', 'Ole', 'Olen',
    'Oley', 'Olie', 'Olin', 'Oliver', 'Ollie', 'Olof', 'Omar', 'Omari',
    'Omarion', 'Omer', 'Oneal', 'Ora', 'Oral', 'Oran', 'Orange', 'Oren',
    'Orie', 'Orin', 'Orion', 'Oris', 'Orla', 'Orland', 'Orlando', 'Orley',
    'Orlin', 'Orlo', 'Orren', 'Orrie', 'Orrin', 'Orris', 'Orson', 'Orval',
    'Orvel', 'Orvil', 'Orville', 'Orvin', 'Orvis', 'Osbaldo', 'Osborn',
    'Osborne', 'Oscar', 'Osie', 'Ossie', 'Osvaldo', 'Oswald', 'Oswaldo',
    'Otha', 'Othel', 'Otho', 'Otis', 'Ott', 'Ottie', 'Ottis', 'Otto', 'Ova',
    'Ovid', 'Ovila', 'Owen', 'Owens', 'Ozell', 'Ozie', 'Ozzie', 'Pablo',
    'Page', 'Palmer', 'Paris', 'Park', 'Parker', 'Parley', 'Parrish',
    'Pascal', 'Pasquale', 'Pat', 'Pate', 'Patric', 'Patrick', 'Paul',
    'Paulo', 'Paxton', 'Payton', 'Pearley', 'Pedro', 'Percival', 'Percy',
    'Perley', 'Pernell', 'Perry', 'Pershing', 'Pete', 'Peter', 'Peyton',
    'Phil', 'Philip', 'Phillip', 'Philo', 'Phoenix', 'Pierce', 'Pierre',
    'Pinkney', 'Pleas', 'Pleasant', 'Ples', 'Plummer', 'Polk', 'Porfirio',
    'Porter', 'Posey', 'Powell', 'Pranav', 'Pratt', 'Prentice', 'Prentiss',
    'Presley', 'Press', 'Preston', 'Price', 'Primus', 'Prince', 'Prosper',
    'Pryor', 'Purl', 'Quentin', 'Quincy', 'Quinn', 'Quint', 'Quinten',
    'Quintin', 'Quinton', 'Rae', 'Raekwon', 'Rafael', 'Rafe', 'Raheem',
    'Rahn', 'Rahsaan', 'Rahul', 'Raiden', 'Rakeem', 'Raleigh', 'Ralph',
    'Ramiro', 'Ramon', 'Ramsey', 'Rance', 'Rand', 'Randal', 'Randall',
    'Randel', 'Randell', 'Randle', 'Randolf', 'Randolph', 'Randy', 'Ransom',
    'Raoul', 'Raphael', 'Raquan', 'Ras', 'Rashaad', 'Rashaan', 'Rashad',
    'Rashawn', 'Rasheed', 'Raul', 'Raven', 'Ray', 'Rayan', 'Rayburn',
    'Rayfield', 'Rayford', 'Raymon', 'Raymond', 'Raymundo', 'Raynard',
    'Rayshawn', 'Reagan', 'Reason', 'Red', 'Redden', 'Redmond', 'Reece',
    'Reed', 'Reese', 'Refugio', 'Regan', 'Reggie', 'Reginal', 'Reginald',
    'Regis', 'Reid', 'Reilly', 'Reinaldo', 'Reinhold', 'Reino', 'Remington',
    'Remy', 'Renaldo', 'Renard', 'Rene', 'Reno', 'Reuben', 'Reubin', 'Rex',
    'Rexford', 'Rey', 'Reyes', 'Reynaldo', 'Reynold', 'Reynolds', 'Rhett',
    'Rhoda', 'Rhys', 'Rian', 'Ricardo', 'Ricci', 'Rice', 'Rich', 'Richard',
    'Richie', 'Richmond', 'Rick', 'Rickey', 'Ricki', 'Rickie', 'Ricky',
    'Rico', 'Ridge', 'Rigoberto', 'Riley', 'Rishi', 'Ritchie', 'River',
    'Rob', 'Robb', 'Robbie', 'Robbin', 'Robby', 'Robert', 'Roberto',
    'Robin', 'Robley', 'Robt', 'Roby', 'Rocco', 'Rock', 'Rocky', 'Rod',
    'Roddy', 'Roderic', 'Roderick', 'Rodger', 'Rodney', 'Rodolfo',
    'Rodrick', 'Rodrigo', 'Roe', 'Roel', 'Rogelio', 'Roger', 'Rogers',
    'Rohan', 'Roland', 'Rolando', 'Rolf', 'Roll', 'Rolla', 'Rolland',
    'Rollie', 'Rollin', 'Rollo', 'Roma', 'Roman', 'Rome', 'Romello',
    'Romeo', 'Romie', 'Ron', 'Ronal', 'Ronald', 'Ronaldo', 'Ronan',
    'Rondal', 'Ronin', 'Ronnie', 'Ronny', 'Roosevelt', 'Rory', 'Rosario',
    'Rosco', 'Roscoe', 'Rosendo', 'Rosevelt', 'Ross', 'Rossie', 'Roswell',
    'Rowan', 'Rowland', 'Roy', 'Royal', 'Royce', 'Rube', 'Ruben', 'Rubin',
    'Ruby', 'Rudolf', 'Rudolfo', 'Rudolph', 'Rudy', 'Rueben', 'Ruel',
    'Ruffin', 'Ruffus', 'Rufus', 'Rupert', 'Rush', 'Russ', 'Russel',
    'Russell', 'Rustin', 'Rusty', 'Rutherford', 'Ryan', 'Ryder', 'Ryker',
    'Rylan', 'Ryland', 'Rylee', 'Ryley', 'Ryne', 'Sabastian', 'Sage',
    'Saint', 'Sal', 'Salomon', 'Salvador', 'Salvatore', 'Sam', 'Samie',
    'Samir', 'Sammie', 'Sammy', 'Sampson', 'Samson', 'Samual', 'Samuel',
    'Sanders', 'Sandy', 'Sanford', 'Santana', 'Santiago', 'Santino',
    'Santo', 'Santos', 'Saul', 'Saverio', 'Savion', 'Savon', 'Sawyer',
    'Schley', 'Schuyler', 'Scot', 'Scott', 'Scottie', 'Scotty', 'Seaborn',
    'Seamus', 'Sean', 'Sebastian', 'Sedrick', 'Seldon', 'Selmer', 'Semaj',
    'Seneca', 'Sergio', 'Seth', 'Severo', 'Severt', 'Seward', 'Seymour',
    'Shad', 'Shade', 'Shafter', 'Shamar', 'Shan', 'Shane', 'Shannon',
    'Shanon', 'Shaquan', 'Shaquille', 'Sharif', 'Sharon', 'Shaun', 'Shawn',
    'Shay', 'Shayne', 'Shea', 'Shedrick', 'Shelby', 'Sheldon', 'Shelley',
    'Shellie', 'Shelly', 'Shelton', 'Shemar', 'Shep', 'Shepherd',
    'Sheridan', 'Sherman', 'Sherrill', 'Sherwin', 'Sherwood', 'Shirley',
    'Shoji', 'Shon', 'Shyheim', 'Sid', 'Sidney', 'Sie', 'Sigmund', 'Sigurd',
    'Silas', 'Silver', 'Silvester', 'Silvio', 'Sim', 'Simeon', 'Simmie',
    'Simon', 'Simpson', 'Sincere', 'Sing', 'Skip', 'Skylar', 'Skyler',
    'Slade', 'Smith', 'Sol', 'Soloman', 'Solomon', 'Solon', 'Son', 'Sonny',
    'Soren', 'Spencer', 'Spenser', 'Spurgeon', 'Squire', 'Stacey', 'Stacy',
    'Stafford', 'Stan', 'Stanford', 'Stanislaus', 'Stanley', 'Stanton',
    'Starling', 'Stefan', 'Stephan', 'Stephanie', 'Stephen', 'Stephon',
    'Sterling', 'Stetson', 'Stevan', 'Steve', 'Steven', 'Stevie', 'Steward',
    'Stewart', 'Stone', 'Stonewall', 'Stoney', 'Storm', 'Stuart',
    'Sullivan', 'Sumner', 'Susie', 'Sydney', 'Syed', 'Sylas', 'Sylvan',
    'Sylvanus', 'Sylvester', 'Tab', 'Tad', 'Taft', 'Tahj', 'Taj', 'Tal',
    'Talan', 'Talen', 'Tallie', 'Talmadge', 'Talmage', 'Talon', 'Tandy',
    'Tanner', 'Tarik', 'Tariq', 'Tate', 'Tatsuo', 'Taurean', 'Taurus',
    'Tavares', 'Tavaris', 'Tavian', 'Tavion', 'Tavon', 'Tayler', 'Taylor',
    'Tayshaun', 'Teagan', 'Ted', 'Teddie', 'Teddy', 'Tegan', 'Telly',
    'Terance', 'Terell', 'Terence', 'Terrance', 'Terrell', 'Terrence',
    'Terrill', 'Terry', 'Tevin', 'Tex', 'Thad', 'Thaddeus', 'Theadore',
    'Thedore', 'Theo', 'Theodis', 'Theodore', 'Theophile', 'Therman',
    'Theron', 'Thomas', 'Thompson', 'Thor', 'Thornton', 'Thorwald', 'Thos',
    'Thurlow', 'Thurman', 'Thurston', 'Tilden', 'Tillman', 'Tilman', 'Tim',
    'Timmie', 'Timmothy', 'Timmy', 'Timothy', 'Tito', 'Titus', 'Tobe',
    'Tobias', 'Tobie', 'Tobin', 'Toby', 'Tod', 'Todd', 'Toivo', 'Tolbert',
    'Tollie', 'Tom', 'Toma', 'Tomas', 'Tomie', 'Tommie', 'Tommy', 'Toney',
    'Tony', 'Torey', 'Toriano', 'Torrance', 'Torrence', 'Torrey', 'Torry',
    'Tory', 'Toshio', 'Toy', 'Trace', 'Tracey', 'Tracy', 'Trae', 'Travis',
    'Travon', 'Trayvon', 'Tre', 'Tremaine', 'Tremayne', 'Trent', 'Trenten',
    'Trenton', 'Trever', 'Trevin', 'Trevion', 'Trevon', 'Trevor', 'Trey',
    'Treyton', 'Treyvon', 'Trinidad', 'Trinity', 'Tripp', 'Tristan',
    'Tristen', 'Tristian', 'Tristin', 'Triston', 'Troy', 'True', 'Trumaine',
    'Truman', 'Trystan', 'Tuan', 'Tucker', 'Turner', 'Ty', 'Tye', 'Tyler',
    'Tylor', 'Tyquan', 'Tyree', 'Tyreek', 'Tyreese', 'Tyrek', 'Tyreke',
    'Tyrel', 'Tyrell', 'Tyrese', 'Tyrik', 'Tyrin', 'Tyriq', 'Tyrique',
    'Tyron', 'Tyrone', 'Tyrus', 'Tyshawn', 'Tyson', 'Ulises', 'Ulysses',
    'Unknown', 'Unnamed', 'Urban', 'Uriah', 'Uriel', 'Urijah', 'Val',
    'Valentin', 'Valentine', 'Valentino', 'Van', 'Vance', 'Vander',
    'Vashon', 'Vaughn', 'Vera', 'Vere', 'Vergil', 'Verl', 'Verle', 'Verlin',
    'Verlon', 'Verlyn', 'Vern', 'Verna', 'Vernal', 'Verne', 'Vernell',
    'Verner', 'Vernie', 'Vernon', 'Vester', 'Vic', 'Vicente', 'Vick',
    'Victor', 'Victoriano', 'Vidal', 'Vince', 'Vincent', 'Vincenzo',
    'Vinson', 'Vinton', 'Virge', 'Virgel', 'Virgie', 'Virgil', 'Virgle',
    'Vito', 'Vollie', 'Volney', 'Von', 'Wade', 'Waino', 'Waldemar', 'Waldo',
    'Walker', 'Wallace', 'Wally', 'Walt', 'Walter', 'Walton', 'Ward',
    'Wardell', 'Warner', 'Warren', 'Wash', 'Washington', 'Watson', 'Watt',
    'Waverly', 'Wayde', 'Wayland', 'Waylon', 'Wayman', 'Waymon', 'Wayne',
    'Weaver', 'Webb', 'Webster', 'Weldon', 'Wellington', 'Wells', 'Welton',
    'Wendel', 'Wendell', 'Wenzel', 'Werner', 'Wes', 'Wesley', 'Wess',
    'West', 'Westin', 'Westley', 'Weston', 'Wheeler', 'Whit', 'Whitney',
    'Wilber', 'Wilbert', 'Wilbur', 'Wilburn', 'Wiley', 'Wilford', 'Wilfred',
    'Wilfredo', 'Wilfrid', 'Wilhelm', 'Wiliam', 'Wilkie', 'Will', 'Willaim',
    'Willam', 'Willard', 'William', 'Williams', 'Willian', 'Williard',
    'Willie', 'Willis', 'Willy', 'Wilmer', 'Wilson', 'Wilton', 'Windell',
    'Winfield', 'Winford', 'Winfred', 'Wing', 'Winifred', 'Winnie',
    'Winston', 'Winthrop', 'Winton', 'Wirt', 'Wm', 'Wong', 'Wood', 'Woodie',
    'Woodroe', 'Woodrow', 'Woodson', 'Woody', 'Worley', 'Worth', 'Wright',
    'Wyatt', 'Wylie', 'Wyman', 'Xander', 'Xavier', 'Xzavier', 'Yaakov',
    'Yadiel', 'Yael', 'Yahir', 'Yair', 'Yancy', 'Yandel', 'Yee', 'Yehuda',
    'Yoel', 'York', 'Yosef', 'Yoshio', 'Young', 'Yurem', 'Yusuf',
    'Zachariah', 'Zachary', 'Zachery', 'Zack', 'Zackary', 'Zackery', 'Zaid',
    'Zaiden', 'Zain', 'Zaire', 'Zakary', 'Zander', 'Zane', 'Zavier',
    'Zavion', 'Zayden', 'Zayne', 'Zeb', 'Zebulon', 'Zechariah', 'Zed',
    'Zeke', 'Zenas', 'Zeno', 'Zigmund', 'Zion', 'Zollie',
}


def _get_oscar_urls(language, shuffled="unshuffled", deduplicated="deduplicated"):
    _BASE_DATA_URL_FORMAT_STR = (
        "https://s3.amazonaws.com/datasets.huggingface.co/oscar/1.0/{shuffled}/{deduplicated}/{language}/")
    _BASE_CHECKSUM_FILE_NAME = "{language}_sha256.txt"
    base_data_url = _BASE_DATA_URL_FORMAT_STR.format(
        shuffled=shuffled, language=language, deduplicated=deduplicated
    )
    checksum_url = base_data_url + _BASE_CHECKSUM_FILE_NAME.format(language=language)
    with fsspec.open(checksum_url, encoding="utf-8") as f:
        data_filenames = [line.decode().split("\t")[0] for line in f if line]
        return [base_data_url + data_filename for data_filename in data_filenames]


def _download_urls(urls):
    for url in urls:
        if not os.path.exists(url.split("/")[-1]):
            os.system(f"wget {url}")


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


def save_enron_line(l2, prev, o):
    """ Process Enron Data """
    l2 = remove_html_tags(l2)
    l2 = l2.split('-----Original Message-----')[0].strip()
    l2 = l2.split('---------------------- Forwarded')[0].strip()
    l2 = l2.split('----- Forwarded')[0].strip()
    l2 = l2.split('---------From:')[0].strip()
    l2 = l2.split('**********************************************************************This')[0].strip()
    l2 = l2.split('**********************************************************************   This')[0].strip()
    l2 = l2.split('******************************************************************This')[0].strip()
    l2 = l2.split('*************************************************This')[0].strip()
    l2 = l2.split('********************************************************************** This')[0].strip()
    l2 = l2.split('--------- Inline attachment follows')[0].strip()
    l2 = l2.split('The information contained in this e-mail message and')[0].strip()
    l2 = l2.split('This message is for the designated recipient')[0].strip()
    l2 = l2.split('***Please be advised')[0].strip()
    l2 = l2.split('*******This message')[0].strip()
    l2 = l2.split('This message (including any attachments) contains')[0].strip()
    l2 = l2.split('*********************************************************')[0].strip()
    l2 = l2.split('_________________________________________________________________Get')[0].strip()
    l2 = l2.split('___________________________________________')[0].strip()
    l2 = l2.split('__________________________________________________ Do')[0].strip()
    l2 = l2.replace("\\\"", " \" ").replace("(", " (").replace(")", ") ").replace("[", " [").replace("]", "] ").replace(
        "?", "? ").replace("!", "! ").replace("? ?", "??").replace("! !", "!!").replace(":", ": ").replace("\t",
                                                                                                           " ").replace(
        "= ", "").replace("=20", "").replace("=90", "").replace("=018", "").replace("=09", "").replace("=3D", "")
    l2 = l2.replace(" s ", " 's ").replace(" ve ", " 've ").replace(" re ", " 're ").replace(" ll ", " 'll ").replace(
        " m ", " 'm ").replace(" t ", " 't ").replace(" d ", " 'd ").replace("  ", " ").replace("  ", " ").replace("  ",
                                                                                                                   " ").replace(
        "  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ").replace(". . .", "...")
    l2 = l2.replace(". yahoo", ".yahoo").replace("www. ", "www.").replace(". htm", ".htm").replace(". co",
                                                                                                   ".co").replace(
        ". org", ".org").replace(". edu", ".edu").replace(". net", ".net").replace(". NET", ".NET").replace(". CO",
                                                                                                            ".CO").replace(
        ". ORG", ".ORG").replace(". EDU", ".EDU").replace(": //", "://")
    l2 = l2.replace(": 0", ":0").replace(": 1", ":1").replace(": 2", ":2").replace(": 3", ":3").replace(": 4",
                                                                                                        ":4").replace(
        ": 5", ":5").replace(": 6", ":6").replace(": 7", ":7").replace(": 8", ":8").replace(": 9", ":9")
    l2 = l2.replace(". url -", ".url - <<").replace(". doc -", ".doc - <<").replace(". pdf -", ".pdf <<").replace(
        ". xls -", ".xls <<").replace(". url", ".url>>").replace(". doc", ".doc>>").replace(". pdf", ".pdf>>").replace(
        ". xls", ".xls>>").replace("<< ", "<<").replace("> >", " ").replace("  ", " ")
    l2 = l2.replace(". URL -", ".URL - <<").replace(". DOC -", ".DOC - <<").replace(". PDF -", ".PDF <<").replace(
        ". XLS -", ".xls <<").replace(". URL", ".URL>>").replace(". DOC", ".DOC>>").replace(". PDF", ".PDF>>").replace(
        ". XLS", ".XLS>>").replace("<< ", "<<").replace("> >", " ").replace("  ", " ")
    l2 = l2.replace("RE:", "").replace("Re:", "").replace("RE: ", "").replace("Re: ", "").replace("Fw: ", "").replace(
        "FW: ", "").replace("FWD: ", "").replace("Fwd: ", "")
    l2 = l2.replace('Importance: High', ':')
    if "Sent:" in l2: return
    l2 = l2.replace("...", "... ").replace("\"\"", " \" ").replace("  ", " ").strip(" -:;[]()\=<>\"").rstrip(".!?")
    l2Arr = l2.split()
    if len(l2Arr) > 3:
        l2 = " ".join(itertools.chain(*[camel_case_split(a) for a in l2Arr]))
        if l2.replace(":", "").replace("[", "").replace("]", "").replace(".", "").replace("!", "").replace("?",
                                                                                                           "").replace(
            ",", "").replace("-", "").replace(";", "").replace(" ", "").lower() in prev: return
        l2 = l2.replace("==", "--")
        l2 = l2.replace("++", "--")
        l2 = l2.replace("*~", "--")
        l2 = l2.replace("||", "--")
        l2 = l2.replace("**", "--")
        l2 = l2.replace("__", "--")
        l2 = l2.replace("##", "--")
        for l3 in l2.split('--'):
            l3 = l3.strip()
            if l3:
                for l4 in l3.split("Subject: "):
                    l4 = l4.strip('=, ')
                    if l4: o.write(l4 + "\tenron\n")
        prev[l2.replace(":", "").replace("[", "").replace("]", "").replace(".", "").replace("!", "").replace("?",
                                                                                                             "").replace(
            ",", "").replace("-", "").replace(";", "").replace(" ", "").lower()] = 1


def has_any(s, lst):
    for l in lst:
        if l in s: return True
    return False


def create_english_dataset(share_dir='/content/drive/Shareddrives/BigScience/'):
    """
    Creates a English dataset from different domains useful for doing PII detection. 
    Domains include enron (email), a subset of civil comments (forum message)
    We do some deduplication and cleanup. 
    We add some PII as augumentation (TBD: some <PERSON> and <ORG> tags added. Need to add some more categories).
    See specific licenses for each dataset. 
  """

    prev = {}
    if not os.path.exists("cleaned_english.tsv"):
        with open("english.tsv", "w", encoding="utf8") as o:

            # https://github.com/reglab/casehold court cases are government works and in the public domain.
            # Annotations and selections are under Apache-2.0 License
            """
      @inproceedings{zhengguha2021,
        title={When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset},
        author={Lucia Zheng and Neel Guha and Brandon R. Anderson and Peter Henderson and Daniel E. Ho},
        year={2021},
        eprint={2104.08671},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        booktitle={Proceedings of the 18th International Conference on Artificial Intelligence and Law},
        publisher={Association for Computing Machinery}
  }      
      """
            with open(f"{share_dir}/casehold.csv", "rb") as f:
                while True:
                    line = f.readline().decode()
                    if not line: break
                    line = line.split(",\"")
                    if len(line) >= 2:
                        line = line[1]
                        line = line.split("(<HOLDING>)")
                        if len(line) == 2:
                            s1, s2 = line
                            s1 = s1.replace("  ", " ").replace("  ", " ")
                            s2 = s2.replace("  ", " ").replace("  ", " ")
                            s2 = ' '.join(s2.split(',')[:-6]).strip(';: ')
                            if s2:
                                o.write(s1 + ' HOLDING: ' + s2 + "\tcasehold\n")
                            else:
                                o.write(s1 + "\tcasehold\n")
                        else:
                            s1 = s1.replace("  ", " ").replace("  ", " ")
                            o.write(s1 + "\tcasehold\n")

            # from https://www.kaggle.com/wcukierski/enron-email-dataset, originally from https://www.cs.cmu.edu/~enron/
            # public data and partially copyrighted works (annotations) used by permission of authors
            """
      Public record data origially published by www.ferc.gov. Subsequent data cleansing by the authors and released
      "as a resource for researchers who are interested in improving current email tools, or understanding how email is currently used". 
      """
            with open(f"{share_dir}/kaggle_enron_emails.csv", "rb") as f:
                in_message = False
                l2 = ""
                while True:
                    l = f.readline()
                    if not l: break
                    l = l.decode().strip()
                    if not in_message and l.startswith("Subject:"):
                        l = l.replace("Subject:", "").strip()
                        if l: l2 = l + ":"
                    if "X-FileName" in l:
                        in_message = True
                        continue
                    elif "Message-ID" in l:
                        save_enron_line(l2, prev, o)
                        l2 = ""
                        in_message = False
                    if in_message:
                        l2 += " " + l

                if l2:
                    save_enron_line(l2, prev, o)

            # https://huggingface.co/datasets/civil_comments - CC0
            """
      @article{DBLP:journals/corr/abs-1903-04561,
  author    = {Daniel Borkan and
               Lucas Dixon and
               Jeffrey Sorensen and
               Nithum Thain and
               Lucy Vasserman},
  title     = {Nuanced Metrics for Measuring Unintended Bias with Real Data for Text
               Classification},
  journal   = {CoRR},
  volume    = {abs/1903.04561},
  year      = {2019},
  url       = {http://arxiv.org/abs/1903.04561},
  archivePrefix = {arXiv},
  eprint    = {1903.04561},
  timestamp = {Sun, 31 Mar 2019 19:01:24 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1903-04561},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
      """
            # An experiment we could try is tagging with another public figure tag called PUBLIC_FIGURE_OPINION
            # so that we could train a model to distinguish between public figure mentions in opinion domains vs non-opinion domains

            dataset = load_dataset("civil_comments")
            for d in (dataset['train'],):
                for idx, data in enumerate(d):
                    score = sum([data[feature] for feature in
                                 ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack',
                                  'sexual_explicit']])
                    l2 = data['text']
                    l2 = l2.replace("\n", " ").replace("  ", " ").replace("  ", " ")
                    l2Arr = l2.split()
                    has_a_name = has_any(first_names, l2Arr)
                    l2_lower = l2.lower()
                    if random.choice([0,
                                      1]) and not has_a_name and "mr." not in l2_lower and "ms." not in l2_lower and "mrs." not in l2_lower and "president" not in l2_lower and "governor" not in l2_lower and "mayor" not in l2_lower:
                        continue
                    if len(l2Arr) > 10 and len(l2Arr) < 50 and (score <= 0.5 or random.randint(0,
                                                                                               10) == 0):  # having too much toxic content may skew the data
                        if has_a_name or "mr." in l2_lower or "ms." in l2_lower or "mrs." in l2_lower or "senator" in l2_lower or "president" in l2_lower or "governor" in l2_lower or "mayor" in l2_lower:
                            o.write(l2 + "\tcivil_comments\n")

        os.system("sort --parallel=32 english.tsv -o english.tsv")

    with open("english_cleaned.tsv", "w", encoding="utf8") as o:
        with open("english.tsv", "rb") as f:
            prev = ""
            while True:
                l = f.readline().decode()
                if not l: break
                l = l.strip()
                l2 = l.replace(":", "").replace("[", "").replace("]", "").replace(".", "").replace("!", "").replace("?",
                                                                                                                    "").replace(
                    ",", "").replace("-", "").replace(";", "").replace(" ", "").lower()
                prev2 = prev.replace(":", "").replace("[", "").replace("]", "").replace(".", "").replace("!",
                                                                                                         "").replace(
                    "?", "").replace(",", "").replace("-", "").replace(";", "").replace(" ", "").lower()
                if prev != "" and (l2 == prev2 or (len(prev) > 10 and len(l) > 10 and prev2[:10] == l2[:10])):
                    if len(l) > len(prev):
                        prev = l
                    continue
                else:
                    if prev:
                        if prev[0] < 'וח':
                            o.write(prev.lstrip(':;.+- ') + "\n")
                    prev = l
        if prev:
            if prev[0] < 'וח':
                o.write(prev.lstrip(':;.+- ') + "\n")

    os.system("sort --parallel=32 english_cleaned.tsv -o english_cleaned.tsv")
    os.system(f"cp english_cleaned.tsv {share_dir}/english_cleaned.tsv")


def check_good_sentence(s, en_lang_cutoff=0.1, junk_ratio=0.5, stopword_check=True):
    # basic dejunk
    s = s.lower().strip()
    if not s: return False
    jr = len([s2 for s2 in s if s2 in junk_dict]) / len(s)
    if jr >= junk_ratio:
        return False
    sArr = [s2.strip("' 0123456789¯_§½¼¾×|†—~\"—±′–'°−{}[]·-\'?,./<>!@#^&*()+-‑=:;`→¶'") for s2 in s.lower().split()]
    if len(sArr) == 0:
        return False
    # stopword check
    if stopword_check and len([s2 for s2 in sArr if s2 in stopwords_en]) / len(sArr) < en_lang_cutoff:
        return False
    else:
        # langid check
        try:
            lang = langid.classify(s)[0]
        except:
            lang = ""
        return lang == "en"


def create_oscar_subset_for_ner():
    with open("pii_oscar.txt", "w", encoding="utf8") as o:
        with open("oscar_sample.txt", "rb") as f:
            while True:
                sent = f.readline().decode()
                if not sent: break
                sent = sent.strip()
                for sent2 in sent.split("<|endoftext|>"):
                    sent2 = sent2.strip()
                    sentArr = sent2.split()
                    if len(sentArr) > 150:
                        # print ("truncating sent", sent)
                        sent2 = " ".join(sentArr[:150])
                    if "Alzheimer's" in sent2 or "Alzheimer" in sent2 or 'heart disease' in sent2 or ' AIDS ' in sent2 or ' HIV ' in sent2 or ' was born ' in sent2 or 'Social Secu' in sent2 or 'socialist' in sent2 or 'republican' in sent2 or 'democrat' in sent2 or 'lower class' in sent2 or ' union ' in sent2 or 'upper class' in sent2 or 'middle class' in sent2 or ' cancer ' in sent2:
                        if 'pussy' not in sent2 and ' cock ' not in sent2:
                            o.write(sent2 + "\n")
        url = _get_oscar_urls("en")[0]
        _download_urls([url])
        file = url.split("/")[-1]
        for sent2 in gzip.open(file):
            sent2 = sent2.decode()
            sent2 = sent2.strip()
            sentArr = sent2.split()
            if len(sentArr) > 150:
                sent2 = " ".join(sentArr[:150])
            # let's just look for disease age, for the bigger set, since terms like "republic" and "democract" are over represented
            if "Alzheimer's" in sent2 or "Alzheimer" in sent2 or 'heart disease' in sent2 or ' AIDS ' in sent2 or ' HIV ' in sent2 or ' was born ' in sent2 or ' cancer ' in sent2:
                if not check_good_sentence(sent2):
                    continue
                if 'pussy' not in sent2 and ' cock ' not in sent2:
                    o.write(sent2 + "\n")
    os.system("sort --parallel=32 pii_oscar.txt -o pii_oscar.txt")


def do_ner(do_casehold=False):
    """ Create English based NER/PII dataset """
    faker_target_lang = Faker(faker_map["en"])
    faker_target_lang.add_provider(person)
    faker_target_lang.add_provider(ssn)
    faker_target_lang.add_provider(address)
    nlp = spacy.load('en_core_web_lg')
    row_id = 0
    with open("pii_en.jsonl", "w", encoding="utf8") as o:

        with open("pii_oscar.txt", "rb") as f:
            prev = ""
            for sent2 in tqdm(f):
                # sent2 = f.readline().decode().strip()
                sent2 = sent2.decode().strip()
                if not sent2: break
                domain = "oscar"
                if sent2 == prev:
                    continue
                if prev:
                    sent3 = prev
                    if sent3[0] in "0123456789":
                        sent3 = sent3.split(" ", 1)[1]
                    sentArr = sent3.split()
                    if sentArr[0].endswith(":"):
                        sentArr = sentArr[:1]
                    if len(sentArr) > 100:
                        sentArr = sentArr[:100]
                    sent3 = " ".join(sentArr)
                    if True:
                        doc = nlp(sent3)
                        entities = list(doc.ents)
                        if [entity for entity in entities if entity.label_ == 'PERSON']:
                            ents = [[entity.text, entity.label_] for entity in entities if
                                    entity.label_ in ('PERSON', 'GPE', 'ORG', 'NORP') and 'http:' not in entity.text]
                            swap = False
                            for label, regex in basic_regex:
                                for x in regex.findall(sent3):
                                    if type(x) != str: continue
                                    ents.append([x, label])
                                    if label in ('GOVT_ID', 'STREET_ADDRESS',):
                                        swap = True
                            if len(ents) > 1 or 'cancer' in sent3 or 'class' in sent3 or 'union' in sent3 or 'democrat' in sent3 or 'republican' in sent3 or 'socialist' in sent3:
                                if len(ents) < 5:
                                    if swap or '@' in sent3 or 'Social Sec' in sent3 or 'password' in sent3:
                                        context = {}
                                        ents2 = []
                                        for item in ents:
                                            if item[1] in ('GOVT_ID', 'STREET_ADDRESS', 'PERSON'):
                                                if item[0] in public_figures:
                                                    item[1] = 'PUBLIC_FIGURE'
                                                else:
                                                    context[item[0]] = context.get(item[0], \
                                                                                   faker_target_lang.name() if " " in
                                                                                                               item[
                                                                                                                   0] and
                                                                                                               item[
                                                                                                                   1] == 'PERSON' else \
                                                                                       faker_target_lang.first_name() if
                                                                                       item[1] == 'PERSON' else \
                                                                                           faker_target_lang.ssn() if
                                                                                           item[1] == 'GOVT_ID' else \
                                                                                               faker_target_lang.address() if
                                                                                               item[
                                                                                                   1] == 'STREET_ADDRESS' else \
                                                                                                   item[0])
                                                    if " " in item[0]:
                                                        context[item[0].split()[0]] = context[item[0]].split()[0]
                                                        context[item[0].split()[-1]] = context[item[0]].split()[-1]
                                                    item[0] = context[item[0]]
                                            ents2.append(item)
                                    else:
                                        ents2 = ents
                                    o.write(json.dumps(
                                        {"text": sent3, "ner": ents2, "domain": domain, "target_lang": "en",
                                         "id": row_id}) + "\n")
                                    row_id += 1
                prev = sent2

        with open("english_cleaned.tsv", "rb") as f:
            # while True:
            #  l = f.readline().decode()
            #  if not l: break
            for l in tqdm(f):
                # sent2 = f.readline().decode().strip()
                l = l.decode().strip()
                l = l.split("\t")
                sent = l[0]
                domain = l[-1].strip()
                if not check_good_sentence(sent):
                    continue
                if not do_casehold and domain == "casehold": continue
                if "Notice No." in sent: continue
                if "TO: ALL COMEX" in sent: continue
                if "TO: All NYMEX" in sent: continue
                if "TO: All New" in sent: continue
                if sent[0] in "0123456789":
                    sent = sent.split(" ", 1)[1]
                sentArr = sent.split()
                if sentArr[0].endswith(":"):
                    sentArr = sentArr[:1]
                if len(sentArr) > 100:
                    sentArr = sentArr[:100]
                sent = " ".join(sentArr)
                doc = nlp(sent)
                entities = list(doc.ents)

                if [entity for entity in entities if entity.label_ == 'PERSON']:
                    ents = [[entity.text, entity.label_] for entity in entities if
                            entity.label_ in ('PERSON', 'GPE', 'ORG', 'NORP') and 'http:' not in entity.text]
                    swap = False
                    for label, regex in basic_regex:
                        for x in regex.findall(sent):
                            if type(x) != str: continue
                            ents.append([x, label])
                            if label in ('GOVT_ID', 'STREET_ADDRESS',):
                                swap = True
                    if len(ents) > 1 and len(ents) < 5:
                        if swap or random.randint(0,
                                                  1) == 0 or '@' in sent or 'Social Sec' in sent or 'password' in sent:
                            context = {}
                            ents2 = []
                            for item in ents:
                                if item[1] in ('GOVT_ID', 'STREET_ADDRESS', 'PERSON'):
                                    if item[0] in public_figures:
                                        item[1] = 'PUBLIC_FIGURE'
                                    else:
                                        context[item[0]] = context.get(item[0], \
                                                                       faker_target_lang.name() if " " in item[0] and
                                                                                                   item[
                                                                                                       1] == 'PERSON' else \
                                                                           faker_target_lang.first_name() if item[
                                                                                                                 1] == 'PERSON' else \
                                                                               faker_target_lang.ssn() if item[
                                                                                                              1] == 'GOVT_ID' else \
                                                                                   faker_target_lang.address() if item[
                                                                                                                      1] == 'STREET_ADDRESS' else \
                                                                                       item[0])
                                        sent = sent.replace(item[0], context[item[0]])
                                        if " " in item[0]:
                                            context[item[0].split()[0]] = context[item[0]].split()[0]
                                            context[item[0].split()[-1]] = context[item[0]].split()[-1]
                                        item[0] = context[item[0]]
                                ents2.append(item)
                        else:
                            ents2 = ents
                        o.write(json.dumps(
                            {"text": sent, "ner": ents2, "domain": domain, "target_lang": "en", "id": row_id}) + "\n")
                        row_id += 1


def pre_translation_steps(target_lang='hi', person_swap=True):
    texts = []
    ner_mappings = []
    row_ids = []
    domains = []
    lbracket = "["
    rbracket = "]"
    if target_lang in ('zh', 'ja', 'ko'):
        lbracket = "[["
        rbracket = "]]"
    faker_target_lang = Faker(faker_map[target_lang])
    faker_target_lang.add_provider(person)
    faker_target_lang.add_provider(geo)

    row_id = -1
    for s in tqdm(open(r"pii_en.jsonl", "rb")):
        s = s.decode().strip()
        if not s: continue
        dat = json.loads(s)
        domain = dat['domain']
        ner = dat['ner']
        text = dat['text']
        if 'id' not in dat:
            row_id += 1
        else:
            row_id = int(dat['id'])
        if 'NYMEX' in text: continue
        ner = [n for n in ner if n[0] not in ("FREE", "’m", 'Social Security')]
        ner2 = []
        if ' cancer ' in text:
            ner2.append(['cancer', 'DISEASE'])
        elif ' HIV ' in text:
            ner2.append(['HIV', 'DISEASE'])
        elif ' AIDS ' in text:
            ner2.append(['AIDS', 'DISEASE'])
        elif "Alzheimer's" in text:
            ner2.append(["Alzheimer's", 'DISEASE'])
        elif "Alzheimer" in text:
            ner2.append(['Alzheimer', 'DISEASE'])
        elif 'heart disease' in text:
            ner2.append(['heart disease', 'DISEASE'])
        elif 'democractic' in text:
            ner2.append(['democractic', 'NORP'])
        elif 'democrats' in text:
            ner2.append(['democrats', 'NORP'])
        elif 'Democrats' in text:
            ner2.append(['Democrats', 'NORP'])
        elif 'democrat' in text:
            ner2.append(['democrat', 'NORP'])
        elif 'Democrat' in text:
            ner2.append(['Democrat', 'NORP'])
        elif 'republican' in text:
            ner2.append(['republican', 'NORP'])
        elif 'republicans' in text:
            ner2.append(['republicans', 'NORP'])
        elif 'Republicans' in text:
            ner2.append(['Republicans', 'NORP'])
        elif 'republican' in text:
            ner2.append(['republican', 'NORP'])
        elif 'Republican' in text:
            ner2.append(['Republican', 'NORP'])
        elif 'socialist' in text:
            ner2.append(['socialist', 'NORP'])
        elif 'Socialist' in text:
            ner2.append(['Socialist', 'NORP'])
        for item in ner:
            itemArr = item[0].split()
            if itemArr[0] in ('Association', 'Society', 'Union') or itemArr[-1] in (
                    'Association', 'Society', 'Party'):
                item[1] = 'NORP'
            if item[0] == 'Enron':
                item[0] = choice(swap_org)
                text = text.replace("Enron", item[0])
                text = text.replace("@enron", '@' + item[0].lower())
                text = text.replace(" enron", ' ' + item[0].lower())
                # print (text)
            if item[0] in public_figures:
                item[1] = 'PUBLIC_FIGURE'
            elif item[0] in country:
                item[1] = 'COUNTRY'
            elif item[0] in NORP:
                item[1] = 'NORP'
            elif '@' in item[0]:
                item[1] = 'EMAIL'
            ner2.append(item)
        # col.extend ([d[0] for d in ner2 if d[1] == 'NORP'])
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
                if item[1] in ('COUNTRY', 'GPE', 'PERSON', 'GOVT_ID', 'STREET_ADDRESS',):  # ORG
                    text = text.replace(" " + item[0] + " ", " *" + str(_idx) + "* ")
                    text = text.replace(" " + item[0] + ",", " *" + str(_idx) + "* ,")
                    text = text.replace(" " + item[0] + "'", " *" + str(_idx) + "*'")
                    text = text.replace(item[0], "*" + str(_idx) + "*")
                    context[item[0]] = context.get(item[0], \
                                                   faker_target_lang.first_name() + " " + random.choice(
                                                       bantu_surnames) if " " in item[0] and
                                                                          item[
                                                                              1] == 'PERSON' and target_lang in (
                                                                              'yo', 'sw') else \
                                                       faker_target_lang.name() if " " in item[
                                                           0] and item[
                                                                                       1] == 'PERSON' else \
                                                           faker_target_lang.first_name() if
                                                           item[
                                                               1] == 'PERSON' else \
                                                               faker_target_lang.country() if
                                                               item[
                                                                   1] == 'COUNTRY' else \
                                                                   faker_target_lang.state() if
                                                                   item[
                                                                       1] == 'GPE' and target_lang != 'zh' else \
                                                                       faker_target_lang.province() if
                                                                       item[
                                                                           1] == 'GPE' and target_lang == 'zh' else \
                                                                           faker_target_lang.ssn() if
                                                                           item[
                                                                               1] == 'GOVT_ID' else \
                                                                               faker_target_lang.address() if
                                                                               item[
                                                                                   1] == 'STREET_ADDRESS' else \
                                                                                   item[
                                                                                       0])

                    ner_mapping["*" + str(_idx) + "*"] = [context[item[0]],
                                                          item[1] if item[
                                                                         1] != 'COUNTRY' else 'GPE']
                    if " " in item[0]:
                        context[item[0].split()[0]] = context[item[0]].split()[0]
                        context[item[0].split()[-1]] = context[item[0]].split()[-1]
                    item[0] = context[item[0]]
                else:
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


def do_translations(texts, target_lang='hi'):
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    model = model.to('cuda').half()
    translations = []
    for src_text_list in tqdm(chunks(texts, 16)):
        batch = tokenizer(src_text_list, return_tensors="pt", padding=True, truncation=True).to('cuda')
        gen = model.generate(**batch, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
        outputs = tokenizer.batch_decode(gen, skip_special_tokens=True)
        translations.extend(outputs)
    return translations


def post_translation_steps(translations, ner_mappings, row_ids, domains, target_lang='hi'):
    rbracket = "]"
    if target_lang in ('zh', 'ja', 'ko'):
        rbracket = "]]"
    with open(f"pii_{target_lang}.jsonl", "w", encoding="utf8") as o:
        for index, trans_text in enumerate(translations):
            ner_found = []
            trans_text = trans_text.lstrip(".")
            trans_text = trans_text.strip()
            trans_text = trans_text.replace("* ", "*").replace(" *", "*").replace('"0*', '*0* ').replace(
                '"1*',
                '*1* ').replace(
                '"2*', '*2* ').replace('"3*', '*3* ').replace('"4*', '*4* '). \
                replace('*0:', '*0* ').replace('*1:', '*1* ').replace('*2:', '*2* ').replace('*3:',
                                                                                             '*3* ').replace(
                '*4:', '*4* '). \
                replace('*0 ', '*0* ').replace('*1 ', '*1* ').replace('*2 ', '*2* ').replace('*3 ',
                                                                                             '*3* ').replace(
                '*4 ', '*4* '). \
                replace(' 0*', '*0* ').replace(' 1*', '*1* ').replace('2 *', '*2* ').replace(' 3*',
                                                                                             '*3* ').replace(
                ' 4*', '*4* ')
            if trans_text.startswith('0') and not trans_text.startswith('0 ['):
                trans_text = '*0* ' + trans_text[1:]
                trans_text = trans_text.replace(". [", " [").replace(".[", " [").replace("  ", " ")
                orig_trans_text = trans_text
                for key, ner_item in ner_mappings[index].items():
                    found = False
                if key in trans_text:
                    found = True
                elif key.replace(" ", "") in trans_text:
                    found = True
                    key = key.replace(" ", "")
                elif key.lstrip('*') in trans_text:
                    found = True
                    key = key.lstrip('*')
                elif key.rstrip('*') in trans_text:
                    found = True
                    key = key.rstrip('*')
                if found:
                    if key[0] == '*':
                        trans_text = trans_text.replace(key, " " + ner_item[0] + " ")
                        ner_found.append(list(ner_item))
                    else:
                        trans_text2 = ""
                        for segment in trans_text.split(key):
                            if rbracket in segment:
                                entity, rest = segment.split(rbracket, 1)
                                entity = entity.strip("[]")
                                ner_found.append([entity, ner_item[1]])
                                trans_text2 += " " + entity + " " + rest
                            else:
                                trans_text2 += " " + segment
                        trans_text = trans_text2.strip()
                trans_text = trans_text.replace("*", " ").replace("[", " ").replace("]", " ").replace(
                    " .",
                    ".").replace(" ,",
                                 ",").replace(
                    "  ", " ").replace("  ", " ").strip()
                if target_lang in ('zh', 'ja', 'ko'):
                    trans_text.replace(" ", "")
                    trans_text.strip('#.')
                if ner_found:
                    j = {'text': trans_text, 'ner': ner_found, 'domain': domains[index],
                         'id': row_ids[index],
                         'lang': target_lang}
                    o.write(json.dumps(j) + "\n")

def create_light_suggestions(target_lang):
  suggestions = []
  for s in tqdm(open(f"pii_{target_lang}.jsonl", "rb")):
    s = s.decode().strip()
    if not s: continue
    dat = json.loads(s)
    ner = dat['ner']
    text = dat['text']
    ner = list(set([tuple(a) for a in ner]))
    for s, label in ner:
      suggestions.extend([{'example_id': _id, 'start': i, 'end':i+len(s), 'tag': label, 'text': s} for i in more_itertools.locate(text, lambda s1: s1==s)])
  json.dump(suggestions, open(f"pii_{target_lang}_suggestions.json", "w", encoding="utf8"))


# create_english_dataset()
# create_oscar_subset_for_ner()
# do_ner()
# do_translation(target_lang="hi")
if __name__ == "__main__":
    if "-target_lang" in sys.argv:
        target_lang = sys.argv[sys.argv.index("-target_lang") + 1]
        if target_lang == "en":
            do_ner()
        else:
            texts, ner_mappings, row_ids, domains = pre_translation_steps(target_lang=target_lang)
            translations = do_translations(texts, target_lang=target_lang)
            post_translation_steps(translations, ner_mappings, row_ids, domains, target_lang=target_lang)
    if "-create_light_suggestions" in sys.argv:
        target_lang = sys.argv[sys.argv.index("-create_light_suggestions") + 1]
        create_light_suggestions(target_lang)
               

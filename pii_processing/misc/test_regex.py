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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir, os.path.pardir)))
import json
import pii_processing

from data_tooling.pii_processing.ontology.ontology_manager import OntologyManager

stopwords_en = set(stopwords.words('english'))

junk_dict = dict([(a, 1) for a in "' 0123456789¯_§½¼¾×|†—~\"—±′–'°−{}[]·-\'?,./<>!@#^&*()+-‑=:;`→¶'"])

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
        domain = dat['domain']
        ner = dat['ner']
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

def post_translation_steps(outfile, translations, original_sentences, ner_mappings, row_ids, domains, target_lang='hi'):
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

# basic idea is to do first a lexicon match, than a spacy match if any, and then do progressive groups of regex.
#TODO: for lang where there is no spacy model, translate to en, do ner in en, do back-trans, match for ner mapping and map to original sentence.
#TODO - do complicated rules, such as PERSON Inc. => ORG
#TODO - calculate fmeasure
#TODO - check for zh working properly
def apply_rules(infile, outfile, rule_base, target_lang, do_ontology_manager=True, do_spacy_if_avail=True, char_before_after_window=10):
  nlp = None
  if do_spacy_if_avail:
    if target_lang == 'en':
      nlp = spacy.load('en_core_web_lg')
    elif target_lang == 'zh':
      nlp = spacy.load('zh_core_web_lg')
    elif target_lang == 'pt':
      nlp = spacy.load('pt_core_news_lg')
    elif target_lang == 'fr':
      nlp = spacy.load('fr_core_news_lg')
    elif target_lang == 'ca':
      nlp = spacy.load('ca_core_news_lg')
  if do_ontology_manager:
    ontology_manager = OntologyManager(target_lang=target_lang)
  else:
    ontology_manager = None
  right = {}
  wrong = {}
  with open(outfile, "w", encoding="utf8") as o:
    for line in tqdm(open(infile, "rb")):
        pred = [] #predicted regex rules ent:label
        d = json.loads(line)
        text = d['text']
        if not text: continue
        predict_ner = d.get('predict_ner',{})
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
        d['predict_ner'] = list(predict_ner.items())
        o.write(json.dumps(d)+"\n")
        for ent, label, rule_id, rule_level in list(set(pred)): 
          if ent not in ner or ner[ent] != label:
            wrong[(ent, label, rule_id, rule_level)] = wrong.get((ent, label, rule_id, rule_level), 0) + 1
          else:
            right[(ent, label, rule_id, rule_level)] = right.get((ent, label, rule_id, rule_level), 0) + 1
  return right, wrong


if __name__ == "__main__":
    rulebase = [([
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
    ]

    initial = target_lang = None
    if "-initial" in sys.argv:
      initial = sys.argv[sys.argv.index("-initial")+1]   
    if "-target_lang" in sys.argv:
      target_lang = sys.argv[sys.argv.index("-target_lang")+1]   
    if target_lang:
      #TODO - load the rulebase dynamically from pii_processing.regex folder a file of the form <initial>_<target_lang>.py
      infile = f"{target_lang}.jsonl"
      outfile = "predicted_"+infile
      right, wrong  = apply_rules(infile, outfile, rulebase, target_lang, char_before_after_window=10)
      print ('right', right)
      print ('wrong', wrong)
      #json.dump(right, open(f"right_regex_{target_lang}.json", "w", encoding="utf8"), indent=1)
      #json.dump(wrong, open(f"wrong_regex_{target_lang}.json", "w", encoding="utf8"), indent=1)

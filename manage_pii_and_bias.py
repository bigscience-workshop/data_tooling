#%%writefile data-tooling/manage_pii_and_bias.py
#NOTE: This code is currently not working and in a state of flux.

#Copyright July 2021 Ontocord LLC. Licensed under Apache v2 https://www.apache.org/licenses/LICENSE-2.0

import copy
import fasttext
from sklearn.cluster import KMeans, MiniBatchKMeans
from time import time
import numpy as np
from datasets import load_dataset
from collections import Counter
from itertools import chain
import os
import glob
from joblib import dump, load
from joblib import Parallel, delayed, parallel_backend
from dask.distributed import Client
import indexed_gzip as igzip
import math
from transformers import AutoTokenizer, AutoModel
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities.engine.operator_config import OperatorConfig
from faker import Faker
from faker.providers import internet
from presidio_analyzer import PatternRecognizer
import langid 
from nltk.corpus import stopwords
import difflib

# TODO: improve the faker name generator to create an even proportion of names by race, ethnic group, social economic status.
# TODO: detect disease, national and ethinic origin, religion, caste, social economic status and political affiliation and do swapping.
# TODO: detect highly sensitive info under GDPR and mask.

from transformers import pipeline

from faker.providers.person.ar_AA import Provider as ArAAProvider
from faker.providers.person.ar_SA import Provider as ArSAProvider
from faker.providers.person.ar_PS import Provider as ArPSProvider
import hashlib, base64

import random
    
class PII_NER_Mixin:  
  mariam_mt = {('aav', 'en'): 'Helsinki-NLP/opus-mt-aav-en', ('aed', 'es'): 'Helsinki-NLP/opus-mt-aed-es', ('af', 'de'): 'Helsinki-NLP/opus-mt-af-de', ('af', 'en'): 'Helsinki-NLP/opus-mt-af-en', ('af', 'eo'): 'Helsinki-NLP/opus-mt-af-eo', ('af', 'es'): 'Helsinki-NLP/opus-mt-af-es', ('af', 'fi'): 'Helsinki-NLP/opus-mt-af-fi', ('af', 'fr'): 'Helsinki-NLP/opus-mt-af-fr', ('af', 'nl'): 'Helsinki-NLP/opus-mt-af-nl', ('af', 'ru'): 'Helsinki-NLP/opus-mt-af-ru', ('af', 'sv'): 'Helsinki-NLP/opus-mt-af-sv', ('afa', 'afa'): 'Helsinki-NLP/opus-mt-afa-afa', ('afa', 'en'): 'Helsinki-NLP/opus-mt-afa-en', ('alv', 'en'): 'Helsinki-NLP/opus-mt-alv-en', ('am', 'sv'): 'Helsinki-NLP/opus-mt-am-sv', ('ar', 'de'): 'Helsinki-NLP/opus-mt-ar-de', ('ar', 'el'): 'Helsinki-NLP/opus-mt-ar-el', ('ar', 'en'): 'Helsinki-NLP/opus-mt-ar-en', ('ar', 'eo'): 'Helsinki-NLP/opus-mt-ar-eo', ('ar', 'es'): 'Helsinki-NLP/opus-mt-ar-es', ('ar', 'fr'): 'Helsinki-NLP/opus-mt-ar-fr', ('ar', 'he'): 'Helsinki-NLP/opus-mt-ar-he', ('ar', 'it'): 'Helsinki-NLP/opus-mt-ar-it', ('ar', 'pl'): 'Helsinki-NLP/opus-mt-ar-pl', ('ar', 'ru'): 'Helsinki-NLP/opus-mt-ar-ru', ('ar', 'tr'): 'Helsinki-NLP/opus-mt-ar-tr', ('art', 'en'): 'Helsinki-NLP/opus-mt-art-en', ('ase', 'de'): 'Helsinki-NLP/opus-mt-ase-de', ('ase', 'en'): 'Helsinki-NLP/opus-mt-ase-en', ('ase', 'es'): 'Helsinki-NLP/opus-mt-ase-es', ('ase', 'fr'): 'Helsinki-NLP/opus-mt-ase-fr', ('ase', 'sv'): 'Helsinki-NLP/opus-mt-ase-sv', ('az', 'en'): 'Helsinki-NLP/opus-mt-az-en', ('az', 'es'): 'Helsinki-NLP/opus-mt-az-es', ('az', 'tr'): 'Helsinki-NLP/opus-mt-az-tr', ('bat', 'en'): 'Helsinki-NLP/opus-mt-bat-en', ('bcl', 'de'): 'Helsinki-NLP/opus-mt-bcl-de', ('bcl', 'en'): 'Helsinki-NLP/opus-mt-bcl-en', ('bcl', 'es'): 'Helsinki-NLP/opus-mt-bcl-es', ('bcl', 'fi'): 'Helsinki-NLP/opus-mt-bcl-fi', ('bcl', 'fr'): 'Helsinki-NLP/opus-mt-bcl-fr', ('bcl', 'sv'): 'Helsinki-NLP/opus-mt-bcl-sv', ('be', 'es'): 'Helsinki-NLP/opus-mt-be-es', ('bem', 'en'): 'Helsinki-NLP/opus-mt-bem-en', ('bem', 'es'): 'Helsinki-NLP/opus-mt-bem-es', ('bem', 'fi'): 'Helsinki-NLP/opus-mt-bem-fi', ('bem', 'fr'): 'Helsinki-NLP/opus-mt-bem-fr', ('bem', 'sv'): 'Helsinki-NLP/opus-mt-bem-sv', ('ber', 'en'): 'Helsinki-NLP/opus-mt-ber-en', ('ber', 'es'): 'Helsinki-NLP/opus-mt-ber-es', ('ber', 'fr'): 'Helsinki-NLP/opus-mt-ber-fr', ('bg', 'de'): 'Helsinki-NLP/opus-mt-bg-de', ('bg', 'en'): 'Helsinki-NLP/opus-mt-bg-en', ('bg', 'eo'): 'Helsinki-NLP/opus-mt-bg-eo', ('bg', 'es'): 'Helsinki-NLP/opus-mt-bg-es', ('bg', 'fi'): 'Helsinki-NLP/opus-mt-bg-fi', ('bg', 'fr'): 'Helsinki-NLP/opus-mt-bg-fr', ('bg', 'it'): 'Helsinki-NLP/opus-mt-bg-it', ('bg', 'ru'): 'Helsinki-NLP/opus-mt-bg-ru', ('bg', 'sv'): 'Helsinki-NLP/opus-mt-bg-sv', ('bg', 'tr'): 'Helsinki-NLP/opus-mt-bg-tr', ('bg', 'uk'): 'Helsinki-NLP/opus-mt-bg-uk', ('bi', 'en'): 'Helsinki-NLP/opus-mt-bi-en', ('bi', 'es'): 'Helsinki-NLP/opus-mt-bi-es', ('bi', 'fr'): 'Helsinki-NLP/opus-mt-bi-fr', ('bi', 'sv'): 'Helsinki-NLP/opus-mt-bi-sv', ('bn', 'en'): 'Helsinki-NLP/opus-mt-bn-en', ('bnt', 'en'): 'Helsinki-NLP/opus-mt-bnt-en', ('bzs', 'en'): 'Helsinki-NLP/opus-mt-bzs-en', ('bzs', 'es'): 'Helsinki-NLP/opus-mt-bzs-es', ('bzs', 'fi'): 'Helsinki-NLP/opus-mt-bzs-fi', ('bzs', 'fr'): 'Helsinki-NLP/opus-mt-bzs-fr', ('bzs', 'sv'): 'Helsinki-NLP/opus-mt-bzs-sv', ('ca', 'de'): 'Helsinki-NLP/opus-mt-ca-de', ('ca', 'en'): 'Helsinki-NLP/opus-mt-ca-en', ('ca', 'es'): 'Helsinki-NLP/opus-mt-ca-es', ('ca', 'fr'): 'Helsinki-NLP/opus-mt-ca-fr', ('ca', 'it'): 'Helsinki-NLP/opus-mt-ca-it', ('ca', 'nl'): 'Helsinki-NLP/opus-mt-ca-nl', ('ca', 'pt'): 'Helsinki-NLP/opus-mt-ca-pt', ('ca', 'uk'): 'Helsinki-NLP/opus-mt-ca-uk', ('cau', 'en'): 'Helsinki-NLP/opus-mt-cau-en', ('ccs', 'en'): 'Helsinki-NLP/opus-mt-ccs-en', ('ceb', 'en'): 'Helsinki-NLP/opus-mt-ceb-en', ('ceb', 'es'): 'Helsinki-NLP/opus-mt-ceb-es', ('ceb', 'fi'): 'Helsinki-NLP/opus-mt-ceb-fi', ('ceb', 'fr'): 'Helsinki-NLP/opus-mt-ceb-fr', ('ceb', 'sv'): 'Helsinki-NLP/opus-mt-ceb-sv', ('cel', 'en'): 'Helsinki-NLP/opus-mt-cel-en', ('chk', 'en'): 'Helsinki-NLP/opus-mt-chk-en', ('chk', 'es'): 'Helsinki-NLP/opus-mt-chk-es', ('chk', 'fr'): 'Helsinki-NLP/opus-mt-chk-fr', ('chk', 'sv'): 'Helsinki-NLP/opus-mt-chk-sv', ('cpf', 'en'): 'Helsinki-NLP/opus-mt-cpf-en', ('cpp', 'cpp'): 'Helsinki-NLP/opus-mt-cpp-cpp', ('cpp', 'en'): 'Helsinki-NLP/opus-mt-cpp-en', ('crs', 'de'): 'Helsinki-NLP/opus-mt-crs-de', ('crs', 'en'): 'Helsinki-NLP/opus-mt-crs-en', ('crs', 'es'): 'Helsinki-NLP/opus-mt-crs-es', ('crs', 'fi'): 'Helsinki-NLP/opus-mt-crs-fi', ('crs', 'fr'): 'Helsinki-NLP/opus-mt-crs-fr', ('crs', 'sv'): 'Helsinki-NLP/opus-mt-crs-sv', ('cs', 'de'): 'Helsinki-NLP/opus-mt-cs-de', ('cs', 'en'): 'Helsinki-NLP/opus-mt-cs-en', ('cs', 'eo'): 'Helsinki-NLP/opus-mt-cs-eo', ('cs', 'fi'): 'Helsinki-NLP/opus-mt-cs-fi', ('cs', 'fr'): 'Helsinki-NLP/opus-mt-cs-fr', ('cs', 'sv'): 'Helsinki-NLP/opus-mt-cs-sv', ('cs', 'uk'): 'Helsinki-NLP/opus-mt-cs-uk', ('csg', 'es'): 'Helsinki-NLP/opus-mt-csg-es', ('csn', 'es'): 'Helsinki-NLP/opus-mt-csn-es', ('cus', 'en'): 'Helsinki-NLP/opus-mt-cus-en', ('cy', 'en'): 'Helsinki-NLP/opus-mt-cy-en', ('da', 'de'): 'Helsinki-NLP/opus-mt-da-de', ('da', 'en'): 'Helsinki-NLP/opus-mt-da-en', ('da', 'eo'): 'Helsinki-NLP/opus-mt-da-eo', ('da', 'es'): 'Helsinki-NLP/opus-mt-da-es', ('da', 'fi'): 'Helsinki-NLP/opus-mt-da-fi', ('da', 'fr'): 'Helsinki-NLP/opus-mt-da-fr', ('da', 'no'): 'Helsinki-NLP/opus-mt-da-no', ('da', 'ru'): 'Helsinki-NLP/opus-mt-da-ru', ('de', 'ZH'): 'Helsinki-NLP/opus-mt-de-ZH', ('de', 'af'): 'Helsinki-NLP/opus-mt-de-af', ('de', 'ar'): 'Helsinki-NLP/opus-mt-de-ar', ('de', 'ase'): 'Helsinki-NLP/opus-mt-de-ase', ('de', 'bcl'): 'Helsinki-NLP/opus-mt-de-bcl', ('de', 'bg'): 'Helsinki-NLP/opus-mt-de-bg', ('de', 'bi'): 'Helsinki-NLP/opus-mt-de-bi', ('de', 'bzs'): 'Helsinki-NLP/opus-mt-de-bzs', ('de', 'ca'): 'Helsinki-NLP/opus-mt-de-ca', ('de', 'crs'): 'Helsinki-NLP/opus-mt-de-crs', ('de', 'cs'): 'Helsinki-NLP/opus-mt-de-cs', ('de', 'da'): 'Helsinki-NLP/opus-mt-de-da', ('de', 'de'): 'Helsinki-NLP/opus-mt-de-de', ('de', 'ee'): 'Helsinki-NLP/opus-mt-de-ee', ('de', 'efi'): 'Helsinki-NLP/opus-mt-de-efi', ('de', 'el'): 'Helsinki-NLP/opus-mt-de-el', ('de', 'en'): 'Helsinki-NLP/opus-mt-de-en', ('de', 'eo'): 'Helsinki-NLP/opus-mt-de-eo', ('de', 'es'): 'Helsinki-NLP/opus-mt-de-es', ('de', 'et'): 'Helsinki-NLP/opus-mt-de-et', ('de', 'eu'): 'Helsinki-NLP/opus-mt-de-eu', ('de', 'fi'): 'Helsinki-NLP/opus-mt-de-fi', ('de', 'fj'): 'Helsinki-NLP/opus-mt-de-fj', ('de', 'fr'): 'Helsinki-NLP/opus-mt-de-fr', ('de', 'gaa'): 'Helsinki-NLP/opus-mt-de-gaa', ('de', 'gil'): 'Helsinki-NLP/opus-mt-de-gil', ('de', 'guw'): 'Helsinki-NLP/opus-mt-de-guw', ('de', 'ha'): 'Helsinki-NLP/opus-mt-de-ha', ('de', 'he'): 'Helsinki-NLP/opus-mt-de-he', ('de', 'hil'): 'Helsinki-NLP/opus-mt-de-hil', ('de', 'ho'): 'Helsinki-NLP/opus-mt-de-ho', ('de', 'hr'): 'Helsinki-NLP/opus-mt-de-hr', ('de', 'ht'): 'Helsinki-NLP/opus-mt-de-ht', ('de', 'hu'): 'Helsinki-NLP/opus-mt-de-hu', ('de', 'ig'): 'Helsinki-NLP/opus-mt-de-ig', ('de', 'ilo'): 'Helsinki-NLP/opus-mt-de-ilo', ('de', 'is'): 'Helsinki-NLP/opus-mt-de-is', ('de', 'iso'): 'Helsinki-NLP/opus-mt-de-iso', ('de', 'it'): 'Helsinki-NLP/opus-mt-de-it', ('de', 'kg'): 'Helsinki-NLP/opus-mt-de-kg', ('de', 'ln'): 'Helsinki-NLP/opus-mt-de-ln', ('de', 'loz'): 'Helsinki-NLP/opus-mt-de-loz', ('de', 'lt'): 'Helsinki-NLP/opus-mt-de-lt', ('de', 'lua'): 'Helsinki-NLP/opus-mt-de-lua', ('de', 'ms'): 'Helsinki-NLP/opus-mt-de-ms', ('de', 'mt'): 'Helsinki-NLP/opus-mt-de-mt', ('de', 'niu'): 'Helsinki-NLP/opus-mt-de-niu', ('de', 'nl'): 'Helsinki-NLP/opus-mt-de-nl', ('de', 'no'): 'Helsinki-NLP/opus-mt-de-no', ('de', 'nso'): 'Helsinki-NLP/opus-mt-de-nso', ('de', 'ny'): 'Helsinki-NLP/opus-mt-de-ny', ('de', 'pag'): 'Helsinki-NLP/opus-mt-de-pag', ('de', 'pap'): 'Helsinki-NLP/opus-mt-de-pap', ('de', 'pis'): 'Helsinki-NLP/opus-mt-de-pis', ('de', 'pl'): 'Helsinki-NLP/opus-mt-de-pl', ('de', 'pon'): 'Helsinki-NLP/opus-mt-de-pon', ('de', 'tl'): 'Helsinki-NLP/opus-mt-de-tl', ('de', 'uk'): 'Helsinki-NLP/opus-mt-de-uk', ('de', 'vi'): 'Helsinki-NLP/opus-mt-de-vi', ('dra', 'en'): 'Helsinki-NLP/opus-mt-dra-en', ('ee', 'de'): 'Helsinki-NLP/opus-mt-ee-de', ('ee', 'en'): 'Helsinki-NLP/opus-mt-ee-en', ('ee', 'es'): 'Helsinki-NLP/opus-mt-ee-es', ('ee', 'fi'): 'Helsinki-NLP/opus-mt-ee-fi', ('ee', 'fr'): 'Helsinki-NLP/opus-mt-ee-fr', ('ee', 'sv'): 'Helsinki-NLP/opus-mt-ee-sv', ('efi', 'de'): 'Helsinki-NLP/opus-mt-efi-de', ('efi', 'en'): 'Helsinki-NLP/opus-mt-efi-en', ('efi', 'fi'): 'Helsinki-NLP/opus-mt-efi-fi', ('efi', 'fr'): 'Helsinki-NLP/opus-mt-efi-fr', ('efi', 'sv'): 'Helsinki-NLP/opus-mt-efi-sv', ('el', 'ar'): 'Helsinki-NLP/opus-mt-el-ar', ('el', 'eo'): 'Helsinki-NLP/opus-mt-el-eo', ('el', 'fi'): 'Helsinki-NLP/opus-mt-el-fi', ('el', 'fr'): 'Helsinki-NLP/opus-mt-el-fr', ('el', 'sv'): 'Helsinki-NLP/opus-mt-el-sv', ('en', 'aav'): 'Helsinki-NLP/opus-mt-en-aav', ('en', 'af'): 'Helsinki-NLP/opus-mt-en-af', ('en', 'afa'): 'Helsinki-NLP/opus-mt-en-afa', ('en', 'alv'): 'Helsinki-NLP/opus-mt-en-alv', ('en', 'ar'): 'Helsinki-NLP/opus-mt-en-ar', ('en', 'az'): 'Helsinki-NLP/opus-mt-en-az', ('en', 'bat'): 'Helsinki-NLP/opus-mt-en-bat', ('en', 'bcl'): 'Helsinki-NLP/opus-mt-en-bcl', ('en', 'bem'): 'Helsinki-NLP/opus-mt-en-bem', ('en', 'ber'): 'Helsinki-NLP/opus-mt-en-ber', ('en', 'bg'): 'Helsinki-NLP/opus-mt-en-bg', ('en', 'bi'): 'Helsinki-NLP/opus-mt-en-bi', ('en', 'bnt'): 'Helsinki-NLP/opus-mt-en-bnt', ('en', 'bzs'): 'Helsinki-NLP/opus-mt-en-bzs', ('en', 'ca'): 'Helsinki-NLP/opus-mt-en-ca', ('en', 'ceb'): 'Helsinki-NLP/opus-mt-en-ceb', ('en', 'cel'): 'Helsinki-NLP/opus-mt-en-cel', ('en', 'chk'): 'Helsinki-NLP/opus-mt-en-chk', ('en', 'cpf'): 'Helsinki-NLP/opus-mt-en-cpf', ('en', 'cpp'): 'Helsinki-NLP/opus-mt-en-cpp', ('en', 'crs'): 'Helsinki-NLP/opus-mt-en-crs', ('en', 'cs'): 'Helsinki-NLP/opus-mt-en-cs', ('en', 'cus'): 'Helsinki-NLP/opus-mt-en-cus', ('en', 'cy'): 'Helsinki-NLP/opus-mt-en-cy', ('en', 'da'): 'Helsinki-NLP/opus-mt-en-da', ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de', ('en', 'dra'): 'Helsinki-NLP/opus-mt-en-dra', ('en', 'ee'): 'Helsinki-NLP/opus-mt-en-ee', ('en', 'efi'): 'Helsinki-NLP/opus-mt-en-efi', ('en', 'el'): 'Helsinki-NLP/opus-mt-en-el', ('en', 'eo'): 'Helsinki-NLP/opus-mt-en-eo', ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es', ('en', 'et'): 'Helsinki-NLP/opus-mt-en-et', ('en', 'eu'): 'Helsinki-NLP/opus-mt-en-eu', ('en', 'euq'): 'Helsinki-NLP/opus-mt-en-euq', ('en', 'fi'): 'Helsinki-NLP/opus-mt-en-fi', ('en', 'fiu'): 'Helsinki-NLP/opus-mt-en-fiu', ('en', 'fj'): 'Helsinki-NLP/opus-mt-en-fj', ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr', ('en', 'ga'): 'Helsinki-NLP/opus-mt-en-ga', ('en', 'gaa'): 'Helsinki-NLP/opus-mt-en-gaa', ('en', 'gem'): 'Helsinki-NLP/opus-mt-en-gem', ('en', 'gil'): 'Helsinki-NLP/opus-mt-en-gil', ('en', 'gl'): 'Helsinki-NLP/opus-mt-en-gl', ('en', 'gmq'): 'Helsinki-NLP/opus-mt-en-gmq', ('en', 'gmw'): 'Helsinki-NLP/opus-mt-en-gmw', ('en', 'grk'): 'Helsinki-NLP/opus-mt-en-grk', ('en', 'guw'): 'Helsinki-NLP/opus-mt-en-guw', ('en', 'gv'): 'Helsinki-NLP/opus-mt-en-gv', ('en', 'ha'): 'Helsinki-NLP/opus-mt-en-ha', ('en', 'he'): 'Helsinki-NLP/opus-mt-en-he', ('en', 'hi'): 'Helsinki-NLP/opus-mt-en-hi', ('en', 'hil'): 'Helsinki-NLP/opus-mt-en-hil', ('en', 'ho'): 'Helsinki-NLP/opus-mt-en-ho', ('en', 'ht'): 'Helsinki-NLP/opus-mt-en-ht', ('en', 'hu'): 'Helsinki-NLP/opus-mt-en-hu', ('en', 'hy'): 'Helsinki-NLP/opus-mt-en-hy', ('en', 'id'): 'Helsinki-NLP/opus-mt-en-id', ('en', 'ig'): 'Helsinki-NLP/opus-mt-en-ig', ('en', 'iir'): 'Helsinki-NLP/opus-mt-en-iir', ('en', 'ilo'): 'Helsinki-NLP/opus-mt-en-ilo', ('en', 'inc'): 'Helsinki-NLP/opus-mt-en-inc', ('en', 'ine'): 'Helsinki-NLP/opus-mt-en-ine', ('en', 'is'): 'Helsinki-NLP/opus-mt-en-is', ('en', 'iso'): 'Helsinki-NLP/opus-mt-en-iso', ('en', 'it'): 'Helsinki-NLP/opus-mt-en-it', ('en', 'itc'): 'Helsinki-NLP/opus-mt-en-itc', ('en', 'jap'): 'Helsinki-NLP/opus-mt-en-jap', ('en', 'kg'): 'Helsinki-NLP/opus-mt-en-kg', ('en', 'kj'): 'Helsinki-NLP/opus-mt-en-kj', ('en', 'kqn'): 'Helsinki-NLP/opus-mt-en-kqn', ('en', 'kwn'): 'Helsinki-NLP/opus-mt-en-kwn', ('en', 'kwy'): 'Helsinki-NLP/opus-mt-en-kwy', ('en', 'lg'): 'Helsinki-NLP/opus-mt-en-lg', ('en', 'ln'): 'Helsinki-NLP/opus-mt-en-ln', ('en', 'loz'): 'Helsinki-NLP/opus-mt-en-loz', ('en', 'lu'): 'Helsinki-NLP/opus-mt-en-lu', ('en', 'lua'): 'Helsinki-NLP/opus-mt-en-lua', ('en', 'lue'): 'Helsinki-NLP/opus-mt-en-lue', ('en', 'lun'): 'Helsinki-NLP/opus-mt-en-lun', ('en', 'luo'): 'Helsinki-NLP/opus-mt-en-luo', ('en', 'lus'): 'Helsinki-NLP/opus-mt-en-lus', ('en', 'map'): 'Helsinki-NLP/opus-mt-en-map', ('en', 'mfe'): 'Helsinki-NLP/opus-mt-en-mfe', ('en', 'mg'): 'Helsinki-NLP/opus-mt-en-mg', ('en', 'mh'): 'Helsinki-NLP/opus-mt-en-mh', ('en', 'mk'): 'Helsinki-NLP/opus-mt-en-mk', ('en', 'mkh'): 'Helsinki-NLP/opus-mt-en-mkh', ('en', 'ml'): 'Helsinki-NLP/opus-mt-en-ml', ('en', 'mos'): 'Helsinki-NLP/opus-mt-en-mos', ('en', 'mr'): 'Helsinki-NLP/opus-mt-en-mr', ('en', 'mt'): 'Helsinki-NLP/opus-mt-en-mt', ('en', 'mul'): 'Helsinki-NLP/opus-mt-en-mul', ('en', 'ng'): 'Helsinki-NLP/opus-mt-en-ng', ('en', 'nic'): 'Helsinki-NLP/opus-mt-en-nic', ('en', 'niu'): 'Helsinki-NLP/opus-mt-en-niu', ('en', 'nl'): 'Helsinki-NLP/opus-mt-en-nl', ('en', 'nso'): 'Helsinki-NLP/opus-mt-en-nso', ('en', 'ny'): 'Helsinki-NLP/opus-mt-en-ny', ('en', 'nyk'): 'Helsinki-NLP/opus-mt-en-nyk', ('en', 'om'): 'Helsinki-NLP/opus-mt-en-om', ('en', 'pag'): 'Helsinki-NLP/opus-mt-en-pag', ('en', 'pap'): 'Helsinki-NLP/opus-mt-en-pap', ('en', 'phi'): 'Helsinki-NLP/opus-mt-en-phi', ('en', 'pis'): 'Helsinki-NLP/opus-mt-en-pis', ('en', 'pon'): 'Helsinki-NLP/opus-mt-en-pon', ('en', 'poz'): 'Helsinki-NLP/opus-mt-en-poz', ('en', 'pqe'): 'Helsinki-NLP/opus-mt-en-pqe', ('en', 'pqw'): 'Helsinki-NLP/opus-mt-en-pqw', ('en', 'rn'): 'Helsinki-NLP/opus-mt-en-rn', ('en', 'rnd'): 'Helsinki-NLP/opus-mt-en-rnd', ('en', 'ro'): 'Helsinki-NLP/opus-mt-en-ro', ('en', 'roa'): 'Helsinki-NLP/opus-mt-en-roa', ('en', 'ru'): 'Helsinki-NLP/opus-mt-en-ru', ('en', 'run'): 'Helsinki-NLP/opus-mt-en-run', ('en', 'rw'): 'Helsinki-NLP/opus-mt-en-rw', ('en', 'sal'): 'Helsinki-NLP/opus-mt-en-sal', ('en', 'sem'): 'Helsinki-NLP/opus-mt-en-sem', ('en', 'sg'): 'Helsinki-NLP/opus-mt-en-sg', ('en', 'sit'): 'Helsinki-NLP/opus-mt-en-sit', ('en', 'sk'): 'Helsinki-NLP/opus-mt-en-sk', ('en', 'sla'): 'Helsinki-NLP/opus-mt-en-sla', ('en', 'sm'): 'Helsinki-NLP/opus-mt-en-sm', ('en', 'sn'): 'Helsinki-NLP/opus-mt-en-sn', ('en', 'sq'): 'Helsinki-NLP/opus-mt-en-sq', ('en', 'ss'): 'Helsinki-NLP/opus-mt-en-ss', ('en', 'st'): 'Helsinki-NLP/opus-mt-en-st', ('en', 'sv'): 'Helsinki-NLP/opus-mt-en-sv', ('en', 'sw'): 'Helsinki-NLP/opus-mt-en-sw', ('en', 'swc'): 'Helsinki-NLP/opus-mt-en-swc', ('en', 'tdt'): 'Helsinki-NLP/opus-mt-en-tdt', ('en', 'ti'): 'Helsinki-NLP/opus-mt-en-ti', ('en', 'tiv'): 'Helsinki-NLP/opus-mt-en-tiv', ('en', 'tl'): 'Helsinki-NLP/opus-mt-en-tl', ('en', 'tll'): 'Helsinki-NLP/opus-mt-en-tll', ('en', 'tn'): 'Helsinki-NLP/opus-mt-en-tn', ('en', 'to'): 'Helsinki-NLP/opus-mt-en-to', ('en', 'toi'): 'Helsinki-NLP/opus-mt-en-toi', ('en', 'tpi'): 'Helsinki-NLP/opus-mt-en-tpi', ('en', 'trk'): 'Helsinki-NLP/opus-mt-en-trk', ('en', 'ts'): 'Helsinki-NLP/opus-mt-en-ts', ('en', 'tut'): 'Helsinki-NLP/opus-mt-en-tut', ('en', 'tvl'): 'Helsinki-NLP/opus-mt-en-tvl', ('en', 'tw'): 'Helsinki-NLP/opus-mt-en-tw', ('en', 'ty'): 'Helsinki-NLP/opus-mt-en-ty', ('en', 'uk'): 'Helsinki-NLP/opus-mt-en-uk', ('en', 'umb'): 'Helsinki-NLP/opus-mt-en-umb', ('en', 'ur'): 'Helsinki-NLP/opus-mt-en-ur', ('en', 'urj'): 'Helsinki-NLP/opus-mt-en-urj', ('en', 'vi'): 'Helsinki-NLP/opus-mt-en-vi', ('en', 'xh'): 'Helsinki-NLP/opus-mt-en-xh', ('en', 'zh'): 'Helsinki-NLP/opus-mt-en-zh', ('en', 'zle'): 'Helsinki-NLP/opus-mt-en-zle', ('en', 'zls'): 'Helsinki-NLP/opus-mt-en-zls', ('en', 'zlw'): 'Helsinki-NLP/opus-mt-en-zlw', ('en_el_es_fi', 'en_el_es_fi'): 'Helsinki-NLP/opus-mt-en_el_es_fi-en_el_es_fi', ('eo', 'af'): 'Helsinki-NLP/opus-mt-eo-af', ('eo', 'bg'): 'Helsinki-NLP/opus-mt-eo-bg', ('eo', 'cs'): 'Helsinki-NLP/opus-mt-eo-cs', ('eo', 'da'): 'Helsinki-NLP/opus-mt-eo-da', ('eo', 'de'): 'Helsinki-NLP/opus-mt-eo-de', ('eo', 'el'): 'Helsinki-NLP/opus-mt-eo-el', ('eo', 'en'): 'Helsinki-NLP/opus-mt-eo-en', ('eo', 'es'): 'Helsinki-NLP/opus-mt-eo-es', ('eo', 'fi'): 'Helsinki-NLP/opus-mt-eo-fi', ('eo', 'fr'): 'Helsinki-NLP/opus-mt-eo-fr', ('eo', 'he'): 'Helsinki-NLP/opus-mt-eo-he', ('eo', 'hu'): 'Helsinki-NLP/opus-mt-eo-hu', ('eo', 'it'): 'Helsinki-NLP/opus-mt-eo-it', ('eo', 'nl'): 'Helsinki-NLP/opus-mt-eo-nl', ('eo', 'pl'): 'Helsinki-NLP/opus-mt-eo-pl', ('eo', 'pt'): 'Helsinki-NLP/opus-mt-eo-pt', ('eo', 'ro'): 'Helsinki-NLP/opus-mt-eo-ro', ('eo', 'ru'): 'Helsinki-NLP/opus-mt-eo-ru', ('eo', 'sh'): 'Helsinki-NLP/opus-mt-eo-sh', ('eo', 'sv'): 'Helsinki-NLP/opus-mt-eo-sv', ('es', 'NORWAY'): 'Helsinki-NLP/opus-mt-es-NORWAY', ('es', 'aed'): 'Helsinki-NLP/opus-mt-es-aed', ('es', 'af'): 'Helsinki-NLP/opus-mt-es-af', ('es', 'ar'): 'Helsinki-NLP/opus-mt-es-ar', ('es', 'ase'): 'Helsinki-NLP/opus-mt-es-ase', ('es', 'bcl'): 'Helsinki-NLP/opus-mt-es-bcl', ('es', 'ber'): 'Helsinki-NLP/opus-mt-es-ber', ('es', 'bg'): 'Helsinki-NLP/opus-mt-es-bg', ('es', 'bi'): 'Helsinki-NLP/opus-mt-es-bi', ('es', 'bzs'): 'Helsinki-NLP/opus-mt-es-bzs', ('es', 'ca'): 'Helsinki-NLP/opus-mt-es-ca', ('es', 'ceb'): 'Helsinki-NLP/opus-mt-es-ceb', ('es', 'crs'): 'Helsinki-NLP/opus-mt-es-crs', ('es', 'cs'): 'Helsinki-NLP/opus-mt-es-cs', ('es', 'csg'): 'Helsinki-NLP/opus-mt-es-csg', ('es', 'csn'): 'Helsinki-NLP/opus-mt-es-csn', ('es', 'da'): 'Helsinki-NLP/opus-mt-es-da', ('es', 'de'): 'Helsinki-NLP/opus-mt-es-de', ('es', 'ee'): 'Helsinki-NLP/opus-mt-es-ee', ('es', 'efi'): 'Helsinki-NLP/opus-mt-es-efi', ('es', 'el'): 'Helsinki-NLP/opus-mt-es-el', ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en', ('es', 'eo'): 'Helsinki-NLP/opus-mt-es-eo', ('es', 'es'): 'Helsinki-NLP/opus-mt-es-es', ('es', 'et'): 'Helsinki-NLP/opus-mt-es-et', ('es', 'eu'): 'Helsinki-NLP/opus-mt-es-eu', ('es', 'fi'): 'Helsinki-NLP/opus-mt-es-fi', ('es', 'fj'): 'Helsinki-NLP/opus-mt-es-fj', ('es', 'fr'): 'Helsinki-NLP/opus-mt-es-fr', ('es', 'gaa'): 'Helsinki-NLP/opus-mt-es-gaa', ('es', 'gil'): 'Helsinki-NLP/opus-mt-es-gil', ('es', 'gl'): 'Helsinki-NLP/opus-mt-es-gl', ('es', 'guw'): 'Helsinki-NLP/opus-mt-es-guw', ('es', 'ha'): 'Helsinki-NLP/opus-mt-es-ha', ('es', 'he'): 'Helsinki-NLP/opus-mt-es-he', ('es', 'hil'): 'Helsinki-NLP/opus-mt-es-hil', ('es', 'ho'): 'Helsinki-NLP/opus-mt-es-ho', ('es', 'hr'): 'Helsinki-NLP/opus-mt-es-hr', ('es', 'ht'): 'Helsinki-NLP/opus-mt-es-ht', ('es', 'id'): 'Helsinki-NLP/opus-mt-es-id', ('es', 'ig'): 'Helsinki-NLP/opus-mt-es-ig', ('es', 'ilo'): 'Helsinki-NLP/opus-mt-es-ilo', ('es', 'is'): 'Helsinki-NLP/opus-mt-es-is', ('es', 'iso'): 'Helsinki-NLP/opus-mt-es-iso', ('es', 'it'): 'Helsinki-NLP/opus-mt-es-it', ('es', 'kg'): 'Helsinki-NLP/opus-mt-es-kg', ('es', 'ln'): 'Helsinki-NLP/opus-mt-es-ln', ('es', 'loz'): 'Helsinki-NLP/opus-mt-es-loz', ('es', 'lt'): 'Helsinki-NLP/opus-mt-es-lt', ('es', 'lua'): 'Helsinki-NLP/opus-mt-es-lua', ('es', 'lus'): 'Helsinki-NLP/opus-mt-es-lus', ('es', 'mfs'): 'Helsinki-NLP/opus-mt-es-mfs', ('es', 'mk'): 'Helsinki-NLP/opus-mt-es-mk', ('es', 'mt'): 'Helsinki-NLP/opus-mt-es-mt', ('es', 'niu'): 'Helsinki-NLP/opus-mt-es-niu', ('es', 'nl'): 'Helsinki-NLP/opus-mt-es-nl', ('es', 'no'): 'Helsinki-NLP/opus-mt-es-no', ('es', 'nso'): 'Helsinki-NLP/opus-mt-es-nso', ('es', 'ny'): 'Helsinki-NLP/opus-mt-es-ny', ('es', 'pag'): 'Helsinki-NLP/opus-mt-es-pag', ('es', 'pap'): 'Helsinki-NLP/opus-mt-es-pap', ('es', 'pis'): 'Helsinki-NLP/opus-mt-es-pis', ('es', 'pl'): 'Helsinki-NLP/opus-mt-es-pl', ('es', 'pon'): 'Helsinki-NLP/opus-mt-es-pon', ('es', 'prl'): 'Helsinki-NLP/opus-mt-es-prl', ('es', 'rn'): 'Helsinki-NLP/opus-mt-es-rn', ('es', 'ro'): 'Helsinki-NLP/opus-mt-es-ro', ('es', 'ru'): 'Helsinki-NLP/opus-mt-es-ru', ('es', 'rw'): 'Helsinki-NLP/opus-mt-es-rw', ('es', 'sg'): 'Helsinki-NLP/opus-mt-es-sg', ('es', 'sl'): 'Helsinki-NLP/opus-mt-es-sl', ('es', 'sm'): 'Helsinki-NLP/opus-mt-es-sm', ('es', 'sn'): 'Helsinki-NLP/opus-mt-es-sn', ('es', 'srn'): 'Helsinki-NLP/opus-mt-es-srn', ('es', 'st'): 'Helsinki-NLP/opus-mt-es-st', ('es', 'swc'): 'Helsinki-NLP/opus-mt-es-swc', ('es', 'tl'): 'Helsinki-NLP/opus-mt-es-tl', ('es', 'tll'): 'Helsinki-NLP/opus-mt-es-tll', ('es', 'tn'): 'Helsinki-NLP/opus-mt-es-tn', ('es', 'to'): 'Helsinki-NLP/opus-mt-es-to', ('es', 'tpi'): 'Helsinki-NLP/opus-mt-es-tpi', ('es', 'tvl'): 'Helsinki-NLP/opus-mt-es-tvl', ('es', 'tw'): 'Helsinki-NLP/opus-mt-es-tw', ('es', 'ty'): 'Helsinki-NLP/opus-mt-es-ty', ('es', 'tzo'): 'Helsinki-NLP/opus-mt-es-tzo', ('es', 'uk'): 'Helsinki-NLP/opus-mt-es-uk', ('es', 've'): 'Helsinki-NLP/opus-mt-es-ve', ('es', 'vi'): 'Helsinki-NLP/opus-mt-es-vi', ('es', 'war'): 'Helsinki-NLP/opus-mt-es-war', ('es', 'wls'): 'Helsinki-NLP/opus-mt-es-wls', ('es', 'xh'): 'Helsinki-NLP/opus-mt-es-xh', ('es', 'yo'): 'Helsinki-NLP/opus-mt-es-yo', ('es', 'yua'): 'Helsinki-NLP/opus-mt-es-yua', ('es', 'zai'): 'Helsinki-NLP/opus-mt-es-zai', ('et', 'de'): 'Helsinki-NLP/opus-mt-et-de', ('et', 'en'): 'Helsinki-NLP/opus-mt-et-en', ('et', 'es'): 'Helsinki-NLP/opus-mt-et-es', ('et', 'fi'): 'Helsinki-NLP/opus-mt-et-fi', ('et', 'fr'): 'Helsinki-NLP/opus-mt-et-fr', ('et', 'ru'): 'Helsinki-NLP/opus-mt-et-ru', ('et', 'sv'): 'Helsinki-NLP/opus-mt-et-sv', ('eu', 'de'): 'Helsinki-NLP/opus-mt-eu-de', ('eu', 'en'): 'Helsinki-NLP/opus-mt-eu-en', ('eu', 'es'): 'Helsinki-NLP/opus-mt-eu-es', ('eu', 'ru'): 'Helsinki-NLP/opus-mt-eu-ru', ('euq', 'en'): 'Helsinki-NLP/opus-mt-euq-en', ('fi', 'NORWAY'): 'Helsinki-NLP/opus-mt-fi-NORWAY', ('fi', 'ZH'): 'Helsinki-NLP/opus-mt-fi-ZH', ('fi', 'af'): 'Helsinki-NLP/opus-mt-fi-af', ('fi', 'bcl'): 'Helsinki-NLP/opus-mt-fi-bcl', ('fi', 'bem'): 'Helsinki-NLP/opus-mt-fi-bem', ('fi', 'bg'): 'Helsinki-NLP/opus-mt-fi-bg', ('fi', 'bzs'): 'Helsinki-NLP/opus-mt-fi-bzs', ('fi', 'ceb'): 'Helsinki-NLP/opus-mt-fi-ceb', ('fi', 'crs'): 'Helsinki-NLP/opus-mt-fi-crs', ('fi', 'cs'): 'Helsinki-NLP/opus-mt-fi-cs', ('fi', 'de'): 'Helsinki-NLP/opus-mt-fi-de', ('fi', 'ee'): 'Helsinki-NLP/opus-mt-fi-ee', ('fi', 'efi'): 'Helsinki-NLP/opus-mt-fi-efi', ('fi', 'el'): 'Helsinki-NLP/opus-mt-fi-el', ('fi', 'en'): 'Helsinki-NLP/opus-mt-fi-en', ('fi', 'eo'): 'Helsinki-NLP/opus-mt-fi-eo', ('fi', 'es'): 'Helsinki-NLP/opus-mt-fi-es', ('fi', 'et'): 'Helsinki-NLP/opus-mt-fi-et', ('fi', 'fi'): 'Helsinki-NLP/opus-mt-fi-fi', ('fi', 'fj'): 'Helsinki-NLP/opus-mt-fi-fj', ('fi', 'fr'): 'Helsinki-NLP/opus-mt-fi-fr', ('fi', 'fse'): 'Helsinki-NLP/opus-mt-fi-fse', ('fi', 'gaa'): 'Helsinki-NLP/opus-mt-fi-gaa', ('fi', 'gil'): 'Helsinki-NLP/opus-mt-fi-gil', ('fi', 'guw'): 'Helsinki-NLP/opus-mt-fi-guw', ('fi', 'ha'): 'Helsinki-NLP/opus-mt-fi-ha', ('fi', 'he'): 'Helsinki-NLP/opus-mt-fi-he', ('fi', 'hil'): 'Helsinki-NLP/opus-mt-fi-hil', ('fi', 'ho'): 'Helsinki-NLP/opus-mt-fi-ho', ('fi', 'hr'): 'Helsinki-NLP/opus-mt-fi-hr', ('fi', 'ht'): 'Helsinki-NLP/opus-mt-fi-ht', ('fi', 'hu'): 'Helsinki-NLP/opus-mt-fi-hu', ('fi', 'id'): 'Helsinki-NLP/opus-mt-fi-id', ('fi', 'ig'): 'Helsinki-NLP/opus-mt-fi-ig', ('fi', 'ilo'): 'Helsinki-NLP/opus-mt-fi-ilo', ('fi', 'is'): 'Helsinki-NLP/opus-mt-fi-is', ('fi', 'iso'): 'Helsinki-NLP/opus-mt-fi-iso', ('fi', 'it'): 'Helsinki-NLP/opus-mt-fi-it', ('fi', 'kg'): 'Helsinki-NLP/opus-mt-fi-kg', ('fi', 'kqn'): 'Helsinki-NLP/opus-mt-fi-kqn', ('fi', 'lg'): 'Helsinki-NLP/opus-mt-fi-lg', ('fi', 'ln'): 'Helsinki-NLP/opus-mt-fi-ln', ('fi', 'lu'): 'Helsinki-NLP/opus-mt-fi-lu', ('fi', 'lua'): 'Helsinki-NLP/opus-mt-fi-lua', ('fi', 'lue'): 'Helsinki-NLP/opus-mt-fi-lue', ('fi', 'lus'): 'Helsinki-NLP/opus-mt-fi-lus', ('fi', 'lv'): 'Helsinki-NLP/opus-mt-fi-lv', ('fi', 'mfe'): 'Helsinki-NLP/opus-mt-fi-mfe', ('fi', 'mg'): 'Helsinki-NLP/opus-mt-fi-mg', ('fi', 'mh'): 'Helsinki-NLP/opus-mt-fi-mh', ('fi', 'mk'): 'Helsinki-NLP/opus-mt-fi-mk', ('fi', 'mos'): 'Helsinki-NLP/opus-mt-fi-mos', ('fi', 'mt'): 'Helsinki-NLP/opus-mt-fi-mt', ('fi', 'niu'): 'Helsinki-NLP/opus-mt-fi-niu', ('fi', 'nl'): 'Helsinki-NLP/opus-mt-fi-nl', ('fi', 'no'): 'Helsinki-NLP/opus-mt-fi-no', ('fi', 'nso'): 'Helsinki-NLP/opus-mt-fi-nso', ('fi', 'ny'): 'Helsinki-NLP/opus-mt-fi-ny', ('fi', 'pag'): 'Helsinki-NLP/opus-mt-fi-pag', ('fi', 'pap'): 'Helsinki-NLP/opus-mt-fi-pap', ('fi', 'pis'): 'Helsinki-NLP/opus-mt-fi-pis', ('fi', 'pon'): 'Helsinki-NLP/opus-mt-fi-pon', ('fi', 'ro'): 'Helsinki-NLP/opus-mt-fi-ro', ('fi', 'ru'): 'Helsinki-NLP/opus-mt-fi-ru', ('fi', 'run'): 'Helsinki-NLP/opus-mt-fi-run', ('fi', 'rw'): 'Helsinki-NLP/opus-mt-fi-rw', ('fi', 'sg'): 'Helsinki-NLP/opus-mt-fi-sg', ('fi', 'sk'): 'Helsinki-NLP/opus-mt-fi-sk', ('fi', 'sl'): 'Helsinki-NLP/opus-mt-fi-sl', ('fi', 'sm'): 'Helsinki-NLP/opus-mt-fi-sm', ('fi', 'sn'): 'Helsinki-NLP/opus-mt-fi-sn', ('fi', 'sq'): 'Helsinki-NLP/opus-mt-fi-sq', ('fi', 'srn'): 'Helsinki-NLP/opus-mt-fi-srn', ('fi', 'st'): 'Helsinki-NLP/opus-mt-fi-st', ('fi', 'sv'): 'Helsinki-NLP/opus-mt-fi-sv', ('fi', 'sw'): 'Helsinki-NLP/opus-mt-fi-sw', ('fi', 'swc'): 'Helsinki-NLP/opus-mt-fi-swc', ('fi', 'tiv'): 'Helsinki-NLP/opus-mt-fi-tiv', ('fi', 'tll'): 'Helsinki-NLP/opus-mt-fi-tll', ('fi', 'tn'): 'Helsinki-NLP/opus-mt-fi-tn', ('fi', 'to'): 'Helsinki-NLP/opus-mt-fi-to', ('fi', 'toi'): 'Helsinki-NLP/opus-mt-fi-toi', ('fi', 'tpi'): 'Helsinki-NLP/opus-mt-fi-tpi', ('fi', 'tr'): 'Helsinki-NLP/opus-mt-fi-tr', ('fi', 'ts'): 'Helsinki-NLP/opus-mt-fi-ts', ('fi', 'tvl'): 'Helsinki-NLP/opus-mt-fi-tvl', ('fi', 'tw'): 'Helsinki-NLP/opus-mt-fi-tw', ('fi', 'ty'): 'Helsinki-NLP/opus-mt-fi-ty', ('fi', 'uk'): 'Helsinki-NLP/opus-mt-fi-uk', ('fi', 've'): 'Helsinki-NLP/opus-mt-fi-ve', ('fi', 'war'): 'Helsinki-NLP/opus-mt-fi-war', ('fi', 'wls'): 'Helsinki-NLP/opus-mt-fi-wls', ('fi', 'xh'): 'Helsinki-NLP/opus-mt-fi-xh', ('fi', 'yap'): 'Helsinki-NLP/opus-mt-fi-yap', ('fi', 'yo'): 'Helsinki-NLP/opus-mt-fi-yo', ('fi', 'zne'): 'Helsinki-NLP/opus-mt-fi-zne', ('fi_nb_no_nn_ru_sv_en', 'SAMI'): 'Helsinki-NLP/opus-mt-fi_nb_no_nn_ru_sv_en-SAMI', ('fiu', 'en'): 'Helsinki-NLP/opus-mt-fiu-en', ('fiu', 'fiu'): 'Helsinki-NLP/opus-mt-fiu-fiu', ('fj', 'en'): 'Helsinki-NLP/opus-mt-fj-en', ('fj', 'fr'): 'Helsinki-NLP/opus-mt-fj-fr', ('fr', 'af'): 'Helsinki-NLP/opus-mt-fr-af', ('fr', 'ar'): 'Helsinki-NLP/opus-mt-fr-ar', ('fr', 'ase'): 'Helsinki-NLP/opus-mt-fr-ase', ('fr', 'bcl'): 'Helsinki-NLP/opus-mt-fr-bcl', ('fr', 'bem'): 'Helsinki-NLP/opus-mt-fr-bem', ('fr', 'ber'): 'Helsinki-NLP/opus-mt-fr-ber', ('fr', 'bg'): 'Helsinki-NLP/opus-mt-fr-bg', ('fr', 'bi'): 'Helsinki-NLP/opus-mt-fr-bi', ('fr', 'bzs'): 'Helsinki-NLP/opus-mt-fr-bzs', ('fr', 'ca'): 'Helsinki-NLP/opus-mt-fr-ca', ('fr', 'ceb'): 'Helsinki-NLP/opus-mt-fr-ceb', ('fr', 'crs'): 'Helsinki-NLP/opus-mt-fr-crs', ('fr', 'de'): 'Helsinki-NLP/opus-mt-fr-de', ('fr', 'ee'): 'Helsinki-NLP/opus-mt-fr-ee', ('fr', 'efi'): 'Helsinki-NLP/opus-mt-fr-efi', ('fr', 'el'): 'Helsinki-NLP/opus-mt-fr-el', ('fr', 'en'): 'Helsinki-NLP/opus-mt-fr-en', ('fr', 'eo'): 'Helsinki-NLP/opus-mt-fr-eo', ('fr', 'es'): 'Helsinki-NLP/opus-mt-fr-es', ('fr', 'fj'): 'Helsinki-NLP/opus-mt-fr-fj', ('fr', 'gaa'): 'Helsinki-NLP/opus-mt-fr-gaa', ('fr', 'gil'): 'Helsinki-NLP/opus-mt-fr-gil', ('fr', 'guw'): 'Helsinki-NLP/opus-mt-fr-guw', ('fr', 'ha'): 'Helsinki-NLP/opus-mt-fr-ha', ('fr', 'he'): 'Helsinki-NLP/opus-mt-fr-he', ('fr', 'hil'): 'Helsinki-NLP/opus-mt-fr-hil', ('fr', 'ho'): 'Helsinki-NLP/opus-mt-fr-ho', ('fr', 'hr'): 'Helsinki-NLP/opus-mt-fr-hr', ('fr', 'ht'): 'Helsinki-NLP/opus-mt-fr-ht', ('fr', 'hu'): 'Helsinki-NLP/opus-mt-fr-hu', ('fr', 'id'): 'Helsinki-NLP/opus-mt-fr-id', ('fr', 'ig'): 'Helsinki-NLP/opus-mt-fr-ig', ('fr', 'ilo'): 'Helsinki-NLP/opus-mt-fr-ilo', ('fr', 'iso'): 'Helsinki-NLP/opus-mt-fr-iso', ('fr', 'kg'): 'Helsinki-NLP/opus-mt-fr-kg', ('fr', 'kqn'): 'Helsinki-NLP/opus-mt-fr-kqn', ('fr', 'kwy'): 'Helsinki-NLP/opus-mt-fr-kwy', ('fr', 'lg'): 'Helsinki-NLP/opus-mt-fr-lg', ('fr', 'ln'): 'Helsinki-NLP/opus-mt-fr-ln', ('fr', 'loz'): 'Helsinki-NLP/opus-mt-fr-loz', ('fr', 'lu'): 'Helsinki-NLP/opus-mt-fr-lu', ('fr', 'lua'): 'Helsinki-NLP/opus-mt-fr-lua', ('fr', 'lue'): 'Helsinki-NLP/opus-mt-fr-lue', ('fr', 'lus'): 'Helsinki-NLP/opus-mt-fr-lus', ('fr', 'mfe'): 'Helsinki-NLP/opus-mt-fr-mfe', ('fr', 'mh'): 'Helsinki-NLP/opus-mt-fr-mh', ('fr', 'mos'): 'Helsinki-NLP/opus-mt-fr-mos', ('fr', 'ms'): 'Helsinki-NLP/opus-mt-fr-ms', ('fr', 'mt'): 'Helsinki-NLP/opus-mt-fr-mt', ('fr', 'niu'): 'Helsinki-NLP/opus-mt-fr-niu', ('fr', 'no'): 'Helsinki-NLP/opus-mt-fr-no', ('fr', 'nso'): 'Helsinki-NLP/opus-mt-fr-nso', ('fr', 'ny'): 'Helsinki-NLP/opus-mt-fr-ny', ('fr', 'pag'): 'Helsinki-NLP/opus-mt-fr-pag', ('fr', 'pap'): 'Helsinki-NLP/opus-mt-fr-pap', ('fr', 'pis'): 'Helsinki-NLP/opus-mt-fr-pis', ('fr', 'pl'): 'Helsinki-NLP/opus-mt-fr-pl', ('fr', 'pon'): 'Helsinki-NLP/opus-mt-fr-pon', ('fr', 'rnd'): 'Helsinki-NLP/opus-mt-fr-rnd', ('fr', 'ro'): 'Helsinki-NLP/opus-mt-fr-ro', ('fr', 'ru'): 'Helsinki-NLP/opus-mt-fr-ru', ('fr', 'run'): 'Helsinki-NLP/opus-mt-fr-run', ('fr', 'rw'): 'Helsinki-NLP/opus-mt-fr-rw', ('fr', 'sg'): 'Helsinki-NLP/opus-mt-fr-sg', ('fr', 'sk'): 'Helsinki-NLP/opus-mt-fr-sk', ('fr', 'sl'): 'Helsinki-NLP/opus-mt-fr-sl', ('fr', 'sm'): 'Helsinki-NLP/opus-mt-fr-sm', ('fr', 'sn'): 'Helsinki-NLP/opus-mt-fr-sn', ('fr', 'srn'): 'Helsinki-NLP/opus-mt-fr-srn', ('fr', 'st'): 'Helsinki-NLP/opus-mt-fr-st', ('fr', 'sv'): 'Helsinki-NLP/opus-mt-fr-sv', ('fr', 'swc'): 'Helsinki-NLP/opus-mt-fr-swc', ('fr', 'tiv'): 'Helsinki-NLP/opus-mt-fr-tiv', ('fr', 'tl'): 'Helsinki-NLP/opus-mt-fr-tl', ('fr', 'tll'): 'Helsinki-NLP/opus-mt-fr-tll', ('fr', 'tn'): 'Helsinki-NLP/opus-mt-fr-tn', ('fr', 'to'): 'Helsinki-NLP/opus-mt-fr-to', ('fr', 'tpi'): 'Helsinki-NLP/opus-mt-fr-tpi', ('fr', 'ts'): 'Helsinki-NLP/opus-mt-fr-ts', ('fr', 'tum'): 'Helsinki-NLP/opus-mt-fr-tum', ('fr', 'tvl'): 'Helsinki-NLP/opus-mt-fr-tvl', ('fr', 'tw'): 'Helsinki-NLP/opus-mt-fr-tw', ('fr', 'ty'): 'Helsinki-NLP/opus-mt-fr-ty', ('fr', 'uk'): 'Helsinki-NLP/opus-mt-fr-uk', ('fr', 've'): 'Helsinki-NLP/opus-mt-fr-ve', ('fr', 'vi'): 'Helsinki-NLP/opus-mt-fr-vi', ('fr', 'war'): 'Helsinki-NLP/opus-mt-fr-war', ('fr', 'wls'): 'Helsinki-NLP/opus-mt-fr-wls', ('fr', 'xh'): 'Helsinki-NLP/opus-mt-fr-xh', ('fr', 'yap'): 'Helsinki-NLP/opus-mt-fr-yap', ('fr', 'yo'): 'Helsinki-NLP/opus-mt-fr-yo', ('fr', 'zne'): 'Helsinki-NLP/opus-mt-fr-zne', ('fse', 'fi'): 'Helsinki-NLP/opus-mt-fse-fi', ('ga', 'en'): 'Helsinki-NLP/opus-mt-ga-en', ('gaa', 'de'): 'Helsinki-NLP/opus-mt-gaa-de', ('gaa', 'en'): 'Helsinki-NLP/opus-mt-gaa-en', ('gaa', 'es'): 'Helsinki-NLP/opus-mt-gaa-es', ('gaa', 'fi'): 'Helsinki-NLP/opus-mt-gaa-fi', ('gaa', 'fr'): 'Helsinki-NLP/opus-mt-gaa-fr', ('gaa', 'sv'): 'Helsinki-NLP/opus-mt-gaa-sv', ('gem', 'en'): 'Helsinki-NLP/opus-mt-gem-en', ('gem', 'gem'): 'Helsinki-NLP/opus-mt-gem-gem', ('gil', 'en'): 'Helsinki-NLP/opus-mt-gil-en', ('gil', 'es'): 'Helsinki-NLP/opus-mt-gil-es', ('gil', 'fi'): 'Helsinki-NLP/opus-mt-gil-fi', ('gil', 'fr'): 'Helsinki-NLP/opus-mt-gil-fr', ('gil', 'sv'): 'Helsinki-NLP/opus-mt-gil-sv', ('gl', 'en'): 'Helsinki-NLP/opus-mt-gl-en', ('gl', 'es'): 'Helsinki-NLP/opus-mt-gl-es', ('gl', 'pt'): 'Helsinki-NLP/opus-mt-gl-pt', ('gmq', 'en'): 'Helsinki-NLP/opus-mt-gmq-en', ('gmq', 'gmq'): 'Helsinki-NLP/opus-mt-gmq-gmq', ('gmw', 'en'): 'Helsinki-NLP/opus-mt-gmw-en', ('gmw', 'gmw'): 'Helsinki-NLP/opus-mt-gmw-gmw', ('grk', 'en'): 'Helsinki-NLP/opus-mt-grk-en', ('guw', 'de'): 'Helsinki-NLP/opus-mt-guw-de', ('guw', 'en'): 'Helsinki-NLP/opus-mt-guw-en', ('guw', 'es'): 'Helsinki-NLP/opus-mt-guw-es', ('guw', 'fi'): 'Helsinki-NLP/opus-mt-guw-fi', ('guw', 'fr'): 'Helsinki-NLP/opus-mt-guw-fr', ('guw', 'sv'): 'Helsinki-NLP/opus-mt-guw-sv', ('gv', 'en'): 'Helsinki-NLP/opus-mt-gv-en', ('ha', 'en'): 'Helsinki-NLP/opus-mt-ha-en', ('ha', 'es'): 'Helsinki-NLP/opus-mt-ha-es', ('ha', 'fi'): 'Helsinki-NLP/opus-mt-ha-fi', ('ha', 'fr'): 'Helsinki-NLP/opus-mt-ha-fr', ('ha', 'sv'): 'Helsinki-NLP/opus-mt-ha-sv', ('he', 'ar'): 'Helsinki-NLP/opus-mt-he-ar', ('he', 'de'): 'Helsinki-NLP/opus-mt-he-de', ('he', 'eo'): 'Helsinki-NLP/opus-mt-he-eo', ('he', 'es'): 'Helsinki-NLP/opus-mt-he-es', ('he', 'fi'): 'Helsinki-NLP/opus-mt-he-fi', ('he', 'fr'): 'Helsinki-NLP/opus-mt-he-fr', ('he', 'it'): 'Helsinki-NLP/opus-mt-he-it', ('he', 'ru'): 'Helsinki-NLP/opus-mt-he-ru', ('he', 'sv'): 'Helsinki-NLP/opus-mt-he-sv', ('he', 'uk'): 'Helsinki-NLP/opus-mt-he-uk', ('hi', 'en'): 'Helsinki-NLP/opus-mt-hi-en', ('hi', 'ur'): 'Helsinki-NLP/opus-mt-hi-ur', ('hil', 'de'): 'Helsinki-NLP/opus-mt-hil-de', ('hil', 'en'): 'Helsinki-NLP/opus-mt-hil-en', ('hil', 'fi'): 'Helsinki-NLP/opus-mt-hil-fi', ('ho', 'en'): 'Helsinki-NLP/opus-mt-ho-en', ('hr', 'es'): 'Helsinki-NLP/opus-mt-hr-es', ('hr', 'fi'): 'Helsinki-NLP/opus-mt-hr-fi', ('hr', 'fr'): 'Helsinki-NLP/opus-mt-hr-fr', ('hr', 'sv'): 'Helsinki-NLP/opus-mt-hr-sv', ('ht', 'en'): 'Helsinki-NLP/opus-mt-ht-en', ('ht', 'es'): 'Helsinki-NLP/opus-mt-ht-es', ('ht', 'fi'): 'Helsinki-NLP/opus-mt-ht-fi', ('ht', 'fr'): 'Helsinki-NLP/opus-mt-ht-fr', ('ht', 'sv'): 'Helsinki-NLP/opus-mt-ht-sv', ('hu', 'de'): 'Helsinki-NLP/opus-mt-hu-de', ('hu', 'en'): 'Helsinki-NLP/opus-mt-hu-en', ('hu', 'eo'): 'Helsinki-NLP/opus-mt-hu-eo', ('hu', 'fi'): 'Helsinki-NLP/opus-mt-hu-fi', ('hu', 'fr'): 'Helsinki-NLP/opus-mt-hu-fr', ('hu', 'sv'): 'Helsinki-NLP/opus-mt-hu-sv', ('hu', 'uk'): 'Helsinki-NLP/opus-mt-hu-uk', ('hy', 'en'): 'Helsinki-NLP/opus-mt-hy-en', ('hy', 'ru'): 'Helsinki-NLP/opus-mt-hy-ru', ('id', 'en'): 'Helsinki-NLP/opus-mt-id-en', ('id', 'es'): 'Helsinki-NLP/opus-mt-id-es', ('id', 'fi'): 'Helsinki-NLP/opus-mt-id-fi', ('id', 'fr'): 'Helsinki-NLP/opus-mt-id-fr', ('id', 'sv'): 'Helsinki-NLP/opus-mt-id-sv', ('ig', 'de'): 'Helsinki-NLP/opus-mt-ig-de', ('ig', 'en'): 'Helsinki-NLP/opus-mt-ig-en', ('ig', 'es'): 'Helsinki-NLP/opus-mt-ig-es', ('ig', 'fi'): 'Helsinki-NLP/opus-mt-ig-fi', ('ig', 'fr'): 'Helsinki-NLP/opus-mt-ig-fr', ('ig', 'sv'): 'Helsinki-NLP/opus-mt-ig-sv', ('iir', 'en'): 'Helsinki-NLP/opus-mt-iir-en', ('iir', 'iir'): 'Helsinki-NLP/opus-mt-iir-iir', ('ilo', 'de'): 'Helsinki-NLP/opus-mt-ilo-de', ('ilo', 'en'): 'Helsinki-NLP/opus-mt-ilo-en', ('ilo', 'es'): 'Helsinki-NLP/opus-mt-ilo-es', ('ilo', 'fi'): 'Helsinki-NLP/opus-mt-ilo-fi', ('ilo', 'sv'): 'Helsinki-NLP/opus-mt-ilo-sv', ('inc', 'en'): 'Helsinki-NLP/opus-mt-inc-en', ('inc', 'inc'): 'Helsinki-NLP/opus-mt-inc-inc', ('ine', 'en'): 'Helsinki-NLP/opus-mt-ine-en', ('ine', 'ine'): 'Helsinki-NLP/opus-mt-ine-ine', ('is', 'de'): 'Helsinki-NLP/opus-mt-is-de', ('is', 'en'): 'Helsinki-NLP/opus-mt-is-en', ('is', 'eo'): 'Helsinki-NLP/opus-mt-is-eo', ('is', 'es'): 'Helsinki-NLP/opus-mt-is-es', ('is', 'fi'): 'Helsinki-NLP/opus-mt-is-fi', ('is', 'fr'): 'Helsinki-NLP/opus-mt-is-fr', ('is', 'it'): 'Helsinki-NLP/opus-mt-is-it', ('is', 'sv'): 'Helsinki-NLP/opus-mt-is-sv', ('iso', 'en'): 'Helsinki-NLP/opus-mt-iso-en', ('iso', 'es'): 'Helsinki-NLP/opus-mt-iso-es', ('iso', 'fi'): 'Helsinki-NLP/opus-mt-iso-fi', ('iso', 'fr'): 'Helsinki-NLP/opus-mt-iso-fr', ('iso', 'sv'): 'Helsinki-NLP/opus-mt-iso-sv', ('it', 'ar'): 'Helsinki-NLP/opus-mt-it-ar', ('it', 'bg'): 'Helsinki-NLP/opus-mt-it-bg', ('it', 'ca'): 'Helsinki-NLP/opus-mt-it-ca', ('it', 'de'): 'Helsinki-NLP/opus-mt-it-de', ('it', 'en'): 'Helsinki-NLP/opus-mt-it-en', ('it', 'eo'): 'Helsinki-NLP/opus-mt-it-eo', ('it', 'es'): 'Helsinki-NLP/opus-mt-it-es', ('it', 'fr'): 'Helsinki-NLP/opus-mt-it-fr', ('it', 'is'): 'Helsinki-NLP/opus-mt-it-is', ('it', 'lt'): 'Helsinki-NLP/opus-mt-it-lt', ('it', 'ms'): 'Helsinki-NLP/opus-mt-it-ms', ('it', 'sv'): 'Helsinki-NLP/opus-mt-it-sv', ('it', 'uk'): 'Helsinki-NLP/opus-mt-it-uk', ('it', 'vi'): 'Helsinki-NLP/opus-mt-it-vi', ('itc', 'en'): 'Helsinki-NLP/opus-mt-itc-en', ('itc', 'itc'): 'Helsinki-NLP/opus-mt-itc-itc', ('ja', 'ar'): 'Helsinki-NLP/opus-mt-ja-ar', ('ja', 'bg'): 'Helsinki-NLP/opus-mt-ja-bg', ('ja', 'da'): 'Helsinki-NLP/opus-mt-ja-da', ('ja', 'de'): 'Helsinki-NLP/opus-mt-ja-de', ('ja', 'en'): 'Helsinki-NLP/opus-mt-ja-en', ('ja', 'es'): 'Helsinki-NLP/opus-mt-ja-es', ('ja', 'fi'): 'Helsinki-NLP/opus-mt-ja-fi', ('ja', 'fr'): 'Helsinki-NLP/opus-mt-ja-fr', ('ja', 'he'): 'Helsinki-NLP/opus-mt-ja-he', ('ja', 'hu'): 'Helsinki-NLP/opus-mt-ja-hu', ('ja', 'it'): 'Helsinki-NLP/opus-mt-ja-it', ('ja', 'ms'): 'Helsinki-NLP/opus-mt-ja-ms', ('ja', 'nl'): 'Helsinki-NLP/opus-mt-ja-nl', ('ja', 'pl'): 'Helsinki-NLP/opus-mt-ja-pl', ('ja', 'pt'): 'Helsinki-NLP/opus-mt-ja-pt', ('ja', 'ru'): 'Helsinki-NLP/opus-mt-ja-ru', ('ja', 'sh'): 'Helsinki-NLP/opus-mt-ja-sh', ('ja', 'sv'): 'Helsinki-NLP/opus-mt-ja-sv', ('ja', 'tr'): 'Helsinki-NLP/opus-mt-ja-tr', ('ja', 'vi'): 'Helsinki-NLP/opus-mt-ja-vi', ('jap', 'en'): 'Helsinki-NLP/opus-mt-jap-en', ('ka', 'en'): 'Helsinki-NLP/opus-mt-ka-en', ('ka', 'ru'): 'Helsinki-NLP/opus-mt-ka-ru', ('kab', 'en'): 'Helsinki-NLP/opus-mt-kab-en', ('kg', 'en'): 'Helsinki-NLP/opus-mt-kg-en', ('kg', 'es'): 'Helsinki-NLP/opus-mt-kg-es', ('kg', 'fr'): 'Helsinki-NLP/opus-mt-kg-fr', ('kg', 'sv'): 'Helsinki-NLP/opus-mt-kg-sv', ('kj', 'en'): 'Helsinki-NLP/opus-mt-kj-en', ('kl', 'en'): 'Helsinki-NLP/opus-mt-kl-en', ('ko', 'de'): 'Helsinki-NLP/opus-mt-ko-de', ('ko', 'en'): 'Helsinki-NLP/opus-mt-ko-en', ('ko', 'es'): 'Helsinki-NLP/opus-mt-ko-es', ('ko', 'fi'): 'Helsinki-NLP/opus-mt-ko-fi', ('ko', 'fr'): 'Helsinki-NLP/opus-mt-ko-fr', ('ko', 'hu'): 'Helsinki-NLP/opus-mt-ko-hu', ('ko', 'ru'): 'Helsinki-NLP/opus-mt-ko-ru', ('ko', 'sv'): 'Helsinki-NLP/opus-mt-ko-sv', ('kqn', 'en'): 'Helsinki-NLP/opus-mt-kqn-en', ('kqn', 'es'): 'Helsinki-NLP/opus-mt-kqn-es', ('kqn', 'fr'): 'Helsinki-NLP/opus-mt-kqn-fr', ('kqn', 'sv'): 'Helsinki-NLP/opus-mt-kqn-sv', ('kwn', 'en'): 'Helsinki-NLP/opus-mt-kwn-en', ('kwy', 'en'): 'Helsinki-NLP/opus-mt-kwy-en', ('kwy', 'fr'): 'Helsinki-NLP/opus-mt-kwy-fr', ('kwy', 'sv'): 'Helsinki-NLP/opus-mt-kwy-sv', ('lg', 'en'): 'Helsinki-NLP/opus-mt-lg-en', ('lg', 'es'): 'Helsinki-NLP/opus-mt-lg-es', ('lg', 'fi'): 'Helsinki-NLP/opus-mt-lg-fi', ('lg', 'fr'): 'Helsinki-NLP/opus-mt-lg-fr', ('lg', 'sv'): 'Helsinki-NLP/opus-mt-lg-sv', ('ln', 'de'): 'Helsinki-NLP/opus-mt-ln-de', ('ln', 'en'): 'Helsinki-NLP/opus-mt-ln-en', ('ln', 'es'): 'Helsinki-NLP/opus-mt-ln-es', ('ln', 'fr'): 'Helsinki-NLP/opus-mt-ln-fr', ('loz', 'de'): 'Helsinki-NLP/opus-mt-loz-de', ('loz', 'en'): 'Helsinki-NLP/opus-mt-loz-en', ('loz', 'es'): 'Helsinki-NLP/opus-mt-loz-es', ('loz', 'fi'): 'Helsinki-NLP/opus-mt-loz-fi', ('loz', 'fr'): 'Helsinki-NLP/opus-mt-loz-fr', ('loz', 'sv'): 'Helsinki-NLP/opus-mt-loz-sv', ('lt', 'de'): 'Helsinki-NLP/opus-mt-lt-de', ('lt', 'eo'): 'Helsinki-NLP/opus-mt-lt-eo', ('lt', 'es'): 'Helsinki-NLP/opus-mt-lt-es', ('lt', 'fr'): 'Helsinki-NLP/opus-mt-lt-fr', ('lt', 'it'): 'Helsinki-NLP/opus-mt-lt-it', ('lt', 'pl'): 'Helsinki-NLP/opus-mt-lt-pl', ('lt', 'ru'): 'Helsinki-NLP/opus-mt-lt-ru', ('lt', 'sv'): 'Helsinki-NLP/opus-mt-lt-sv', ('lt', 'tr'): 'Helsinki-NLP/opus-mt-lt-tr', ('lu', 'en'): 'Helsinki-NLP/opus-mt-lu-en', ('lu', 'es'): 'Helsinki-NLP/opus-mt-lu-es', ('lu', 'fi'): 'Helsinki-NLP/opus-mt-lu-fi', ('lu', 'fr'): 'Helsinki-NLP/opus-mt-lu-fr', ('lu', 'sv'): 'Helsinki-NLP/opus-mt-lu-sv', ('lua', 'en'): 'Helsinki-NLP/opus-mt-lua-en', ('lua', 'es'): 'Helsinki-NLP/opus-mt-lua-es', ('lua', 'fi'): 'Helsinki-NLP/opus-mt-lua-fi', ('lua', 'fr'): 'Helsinki-NLP/opus-mt-lua-fr', ('lua', 'sv'): 'Helsinki-NLP/opus-mt-lua-sv', ('lue', 'en'): 'Helsinki-NLP/opus-mt-lue-en', ('lue', 'es'): 'Helsinki-NLP/opus-mt-lue-es', ('lue', 'fi'): 'Helsinki-NLP/opus-mt-lue-fi', ('lue', 'fr'): 'Helsinki-NLP/opus-mt-lue-fr', ('lue', 'sv'): 'Helsinki-NLP/opus-mt-lue-sv', ('lun', 'en'): 'Helsinki-NLP/opus-mt-lun-en', ('luo', 'en'): 'Helsinki-NLP/opus-mt-luo-en', ('lus', 'en'): 'Helsinki-NLP/opus-mt-lus-en', ('lus', 'es'): 'Helsinki-NLP/opus-mt-lus-es', ('lus', 'fi'): 'Helsinki-NLP/opus-mt-lus-fi', ('lus', 'fr'): 'Helsinki-NLP/opus-mt-lus-fr', ('lus', 'sv'): 'Helsinki-NLP/opus-mt-lus-sv', ('lv', 'en'): 'Helsinki-NLP/opus-mt-lv-en', ('lv', 'es'): 'Helsinki-NLP/opus-mt-lv-es', ('lv', 'fi'): 'Helsinki-NLP/opus-mt-lv-fi', ('lv', 'fr'): 'Helsinki-NLP/opus-mt-lv-fr', ('lv', 'ru'): 'Helsinki-NLP/opus-mt-lv-ru', ('lv', 'sv'): 'Helsinki-NLP/opus-mt-lv-sv', ('mfe', 'en'): 'Helsinki-NLP/opus-mt-mfe-en', ('mfe', 'es'): 'Helsinki-NLP/opus-mt-mfe-es', ('mfs', 'es'): 'Helsinki-NLP/opus-mt-mfs-es', ('mg', 'en'): 'Helsinki-NLP/opus-mt-mg-en', ('mg', 'es'): 'Helsinki-NLP/opus-mt-mg-es', ('mh', 'en'): 'Helsinki-NLP/opus-mt-mh-en', ('mh', 'es'): 'Helsinki-NLP/opus-mt-mh-es', ('mh', 'fi'): 'Helsinki-NLP/opus-mt-mh-fi', ('mk', 'en'): 'Helsinki-NLP/opus-mt-mk-en', ('mk', 'es'): 'Helsinki-NLP/opus-mt-mk-es', ('mk', 'fi'): 'Helsinki-NLP/opus-mt-mk-fi', ('mk', 'fr'): 'Helsinki-NLP/opus-mt-mk-fr', ('mkh', 'en'): 'Helsinki-NLP/opus-mt-mkh-en', ('ml', 'en'): 'Helsinki-NLP/opus-mt-ml-en', ('mos', 'en'): 'Helsinki-NLP/opus-mt-mos-en', ('mr', 'en'): 'Helsinki-NLP/opus-mt-mr-en', ('ms', 'de'): 'Helsinki-NLP/opus-mt-ms-de', ('ms', 'fr'): 'Helsinki-NLP/opus-mt-ms-fr', ('ms', 'it'): 'Helsinki-NLP/opus-mt-ms-it', ('ms', 'ms'): 'Helsinki-NLP/opus-mt-ms-ms', ('mt', 'en'): 'Helsinki-NLP/opus-mt-mt-en', ('mt', 'es'): 'Helsinki-NLP/opus-mt-mt-es', ('mt', 'fi'): 'Helsinki-NLP/opus-mt-mt-fi', ('mt', 'fr'): 'Helsinki-NLP/opus-mt-mt-fr', ('mt', 'sv'): 'Helsinki-NLP/opus-mt-mt-sv', ('mul', 'en'): 'Helsinki-NLP/opus-mt-mul-en', ('ng', 'en'): 'Helsinki-NLP/opus-mt-ng-en', ('nic', 'en'): 'Helsinki-NLP/opus-mt-nic-en', ('niu', 'de'): 'Helsinki-NLP/opus-mt-niu-de', ('niu', 'en'): 'Helsinki-NLP/opus-mt-niu-en', ('niu', 'es'): 'Helsinki-NLP/opus-mt-niu-es', ('niu', 'fi'): 'Helsinki-NLP/opus-mt-niu-fi', ('niu', 'fr'): 'Helsinki-NLP/opus-mt-niu-fr', ('niu', 'sv'): 'Helsinki-NLP/opus-mt-niu-sv', ('nl', 'af'): 'Helsinki-NLP/opus-mt-nl-af', ('nl', 'ca'): 'Helsinki-NLP/opus-mt-nl-ca', ('nl', 'en'): 'Helsinki-NLP/opus-mt-nl-en', ('nl', 'eo'): 'Helsinki-NLP/opus-mt-nl-eo', ('nl', 'es'): 'Helsinki-NLP/opus-mt-nl-es', ('nl', 'fi'): 'Helsinki-NLP/opus-mt-nl-fi', ('nl', 'fr'): 'Helsinki-NLP/opus-mt-nl-fr', ('nl', 'no'): 'Helsinki-NLP/opus-mt-nl-no', ('nl', 'sv'): 'Helsinki-NLP/opus-mt-nl-sv', ('nl', 'uk'): 'Helsinki-NLP/opus-mt-nl-uk', ('no', 'da'): 'Helsinki-NLP/opus-mt-no-da', ('no', 'de'): 'Helsinki-NLP/opus-mt-no-de', ('no', 'es'): 'Helsinki-NLP/opus-mt-no-es', ('no', 'fi'): 'Helsinki-NLP/opus-mt-no-fi', ('no', 'fr'): 'Helsinki-NLP/opus-mt-no-fr', ('no', 'nl'): 'Helsinki-NLP/opus-mt-no-nl', ('no', 'no'): 'Helsinki-NLP/opus-mt-no-no', ('no', 'pl'): 'Helsinki-NLP/opus-mt-no-pl', ('no', 'ru'): 'Helsinki-NLP/opus-mt-no-ru', ('no', 'sv'): 'Helsinki-NLP/opus-mt-no-sv', ('no', 'uk'): 'Helsinki-NLP/opus-mt-no-uk', ('nso', 'de'): 'Helsinki-NLP/opus-mt-nso-de', ('nso', 'en'): 'Helsinki-NLP/opus-mt-nso-en', ('nso', 'es'): 'Helsinki-NLP/opus-mt-nso-es', ('nso', 'fi'): 'Helsinki-NLP/opus-mt-nso-fi', ('nso', 'fr'): 'Helsinki-NLP/opus-mt-nso-fr', ('nso', 'sv'): 'Helsinki-NLP/opus-mt-nso-sv', ('ny', 'de'): 'Helsinki-NLP/opus-mt-ny-de', ('ny', 'en'): 'Helsinki-NLP/opus-mt-ny-en', ('ny', 'es'): 'Helsinki-NLP/opus-mt-ny-es', ('nyk', 'en'): 'Helsinki-NLP/opus-mt-nyk-en', ('om', 'en'): 'Helsinki-NLP/opus-mt-om-en', ('pa', 'en'): 'Helsinki-NLP/opus-mt-pa-en', ('pag', 'de'): 'Helsinki-NLP/opus-mt-pag-de', ('pag', 'en'): 'Helsinki-NLP/opus-mt-pag-en', ('pag', 'es'): 'Helsinki-NLP/opus-mt-pag-es', ('pag', 'fi'): 'Helsinki-NLP/opus-mt-pag-fi', ('pag', 'sv'): 'Helsinki-NLP/opus-mt-pag-sv', ('pap', 'de'): 'Helsinki-NLP/opus-mt-pap-de', ('pap', 'en'): 'Helsinki-NLP/opus-mt-pap-en', ('pap', 'es'): 'Helsinki-NLP/opus-mt-pap-es', ('pap', 'fi'): 'Helsinki-NLP/opus-mt-pap-fi', ('pap', 'fr'): 'Helsinki-NLP/opus-mt-pap-fr', ('phi', 'en'): 'Helsinki-NLP/opus-mt-phi-en', ('pis', 'en'): 'Helsinki-NLP/opus-mt-pis-en', ('pis', 'es'): 'Helsinki-NLP/opus-mt-pis-es', ('pis', 'fi'): 'Helsinki-NLP/opus-mt-pis-fi', ('pis', 'fr'): 'Helsinki-NLP/opus-mt-pis-fr', ('pis', 'sv'): 'Helsinki-NLP/opus-mt-pis-sv', ('pl', 'ar'): 'Helsinki-NLP/opus-mt-pl-ar', ('pl', 'de'): 'Helsinki-NLP/opus-mt-pl-de', ('pl', 'en'): 'Helsinki-NLP/opus-mt-pl-en', ('pl', 'eo'): 'Helsinki-NLP/opus-mt-pl-eo', ('pl', 'es'): 'Helsinki-NLP/opus-mt-pl-es', ('pl', 'fr'): 'Helsinki-NLP/opus-mt-pl-fr', ('pl', 'lt'): 'Helsinki-NLP/opus-mt-pl-lt', ('pl', 'no'): 'Helsinki-NLP/opus-mt-pl-no', ('pl', 'sv'): 'Helsinki-NLP/opus-mt-pl-sv', ('pl', 'uk'): 'Helsinki-NLP/opus-mt-pl-uk', ('pon', 'en'): 'Helsinki-NLP/opus-mt-pon-en', ('pon', 'es'): 'Helsinki-NLP/opus-mt-pon-es', ('pon', 'fi'): 'Helsinki-NLP/opus-mt-pon-fi', ('pon', 'fr'): 'Helsinki-NLP/opus-mt-pon-fr', ('pon', 'sv'): 'Helsinki-NLP/opus-mt-pon-sv', ('pqe', 'en'): 'Helsinki-NLP/opus-mt-pqe-en', ('prl', 'es'): 'Helsinki-NLP/opus-mt-prl-es', ('pt', 'ca'): 'Helsinki-NLP/opus-mt-pt-ca', ('pt', 'eo'): 'Helsinki-NLP/opus-mt-pt-eo', ('pt', 'gl'): 'Helsinki-NLP/opus-mt-pt-gl', ('pt', 'tl'): 'Helsinki-NLP/opus-mt-pt-tl', ('pt', 'uk'): 'Helsinki-NLP/opus-mt-pt-uk', ('rn', 'de'): 'Helsinki-NLP/opus-mt-rn-de', ('rn', 'en'): 'Helsinki-NLP/opus-mt-rn-en', ('rn', 'es'): 'Helsinki-NLP/opus-mt-rn-es', ('rn', 'fr'): 'Helsinki-NLP/opus-mt-rn-fr', ('rn', 'ru'): 'Helsinki-NLP/opus-mt-rn-ru', ('rnd', 'en'): 'Helsinki-NLP/opus-mt-rnd-en', ('rnd', 'fr'): 'Helsinki-NLP/opus-mt-rnd-fr', ('rnd', 'sv'): 'Helsinki-NLP/opus-mt-rnd-sv', ('ro', 'eo'): 'Helsinki-NLP/opus-mt-ro-eo', ('ro', 'fi'): 'Helsinki-NLP/opus-mt-ro-fi', ('ro', 'fr'): 'Helsinki-NLP/opus-mt-ro-fr', ('ro', 'sv'): 'Helsinki-NLP/opus-mt-ro-sv', ('roa', 'en'): 'Helsinki-NLP/opus-mt-roa-en', ('ru', 'af'): 'Helsinki-NLP/opus-mt-ru-af', ('ru', 'ar'): 'Helsinki-NLP/opus-mt-ru-ar', ('ru', 'bg'): 'Helsinki-NLP/opus-mt-ru-bg', ('ru', 'da'): 'Helsinki-NLP/opus-mt-ru-da', ('ru', 'en'): 'Helsinki-NLP/opus-mt-ru-en', ('ru', 'eo'): 'Helsinki-NLP/opus-mt-ru-eo', ('ru', 'es'): 'Helsinki-NLP/opus-mt-ru-es', ('ru', 'et'): 'Helsinki-NLP/opus-mt-ru-et', ('ru', 'eu'): 'Helsinki-NLP/opus-mt-ru-eu', ('ru', 'fi'): 'Helsinki-NLP/opus-mt-ru-fi', ('ru', 'fr'): 'Helsinki-NLP/opus-mt-ru-fr', ('ru', 'he'): 'Helsinki-NLP/opus-mt-ru-he', ('ru', 'hy'): 'Helsinki-NLP/opus-mt-ru-hy', ('ru', 'lt'): 'Helsinki-NLP/opus-mt-ru-lt', ('ru', 'lv'): 'Helsinki-NLP/opus-mt-ru-lv', ('ru', 'no'): 'Helsinki-NLP/opus-mt-ru-no', ('ru', 'sl'): 'Helsinki-NLP/opus-mt-ru-sl', ('ru', 'sv'): 'Helsinki-NLP/opus-mt-ru-sv', ('ru', 'uk'): 'Helsinki-NLP/opus-mt-ru-uk', ('ru', 'vi'): 'Helsinki-NLP/opus-mt-ru-vi', ('run', 'en'): 'Helsinki-NLP/opus-mt-run-en', ('run', 'es'): 'Helsinki-NLP/opus-mt-run-es', ('run', 'sv'): 'Helsinki-NLP/opus-mt-run-sv', ('rw', 'en'): 'Helsinki-NLP/opus-mt-rw-en', ('rw', 'es'): 'Helsinki-NLP/opus-mt-rw-es', ('rw', 'fr'): 'Helsinki-NLP/opus-mt-rw-fr', ('rw', 'sv'): 'Helsinki-NLP/opus-mt-rw-sv', ('sal', 'en'): 'Helsinki-NLP/opus-mt-sal-en', ('sem', 'en'): 'Helsinki-NLP/opus-mt-sem-en', ('sem', 'sem'): 'Helsinki-NLP/opus-mt-sem-sem', ('sg', 'en'): 'Helsinki-NLP/opus-mt-sg-en', ('sg', 'es'): 'Helsinki-NLP/opus-mt-sg-es', ('sg', 'fi'): 'Helsinki-NLP/opus-mt-sg-fi', ('sg', 'fr'): 'Helsinki-NLP/opus-mt-sg-fr', ('sg', 'sv'): 'Helsinki-NLP/opus-mt-sg-sv', ('sh', 'eo'): 'Helsinki-NLP/opus-mt-sh-eo', ('sh', 'uk'): 'Helsinki-NLP/opus-mt-sh-uk', ('sk', 'en'): 'Helsinki-NLP/opus-mt-sk-en', ('sk', 'es'): 'Helsinki-NLP/opus-mt-sk-es', ('sk', 'fi'): 'Helsinki-NLP/opus-mt-sk-fi', ('sk', 'fr'): 'Helsinki-NLP/opus-mt-sk-fr', ('sk', 'sv'): 'Helsinki-NLP/opus-mt-sk-sv', ('sl', 'es'): 'Helsinki-NLP/opus-mt-sl-es', ('sl', 'fi'): 'Helsinki-NLP/opus-mt-sl-fi', ('sl', 'fr'): 'Helsinki-NLP/opus-mt-sl-fr', ('sl', 'ru'): 'Helsinki-NLP/opus-mt-sl-ru', ('sl', 'sv'): 'Helsinki-NLP/opus-mt-sl-sv', ('sl', 'uk'): 'Helsinki-NLP/opus-mt-sl-uk', ('sla', 'en'): 'Helsinki-NLP/opus-mt-sla-en', ('sla', 'sla'): 'Helsinki-NLP/opus-mt-sla-sla', ('sm', 'en'): 'Helsinki-NLP/opus-mt-sm-en', ('sm', 'es'): 'Helsinki-NLP/opus-mt-sm-es', ('sm', 'fr'): 'Helsinki-NLP/opus-mt-sm-fr', ('sn', 'en'): 'Helsinki-NLP/opus-mt-sn-en', ('sn', 'es'): 'Helsinki-NLP/opus-mt-sn-es', ('sn', 'fr'): 'Helsinki-NLP/opus-mt-sn-fr', ('sn', 'sv'): 'Helsinki-NLP/opus-mt-sn-sv', ('sq', 'en'): 'Helsinki-NLP/opus-mt-sq-en', ('sq', 'es'): 'Helsinki-NLP/opus-mt-sq-es', ('sq', 'sv'): 'Helsinki-NLP/opus-mt-sq-sv', ('srn', 'en'): 'Helsinki-NLP/opus-mt-srn-en', ('srn', 'es'): 'Helsinki-NLP/opus-mt-srn-es', ('srn', 'fr'): 'Helsinki-NLP/opus-mt-srn-fr', ('srn', 'sv'): 'Helsinki-NLP/opus-mt-srn-sv', ('ss', 'en'): 'Helsinki-NLP/opus-mt-ss-en', ('ssp', 'es'): 'Helsinki-NLP/opus-mt-ssp-es', ('st', 'en'): 'Helsinki-NLP/opus-mt-st-en', ('st', 'es'): 'Helsinki-NLP/opus-mt-st-es', ('st', 'fi'): 'Helsinki-NLP/opus-mt-st-fi', ('st', 'fr'): 'Helsinki-NLP/opus-mt-st-fr', ('st', 'sv'): 'Helsinki-NLP/opus-mt-st-sv', ('sv', 'NORWAY'): 'Helsinki-NLP/opus-mt-sv-NORWAY', ('sv', 'ZH'): 'Helsinki-NLP/opus-mt-sv-ZH', ('sv', 'af'): 'Helsinki-NLP/opus-mt-sv-af', ('sv', 'ase'): 'Helsinki-NLP/opus-mt-sv-ase', ('sv', 'bcl'): 'Helsinki-NLP/opus-mt-sv-bcl', ('sv', 'bem'): 'Helsinki-NLP/opus-mt-sv-bem', ('sv', 'bg'): 'Helsinki-NLP/opus-mt-sv-bg', ('sv', 'bi'): 'Helsinki-NLP/opus-mt-sv-bi', ('sv', 'bzs'): 'Helsinki-NLP/opus-mt-sv-bzs', ('sv', 'ceb'): 'Helsinki-NLP/opus-mt-sv-ceb', ('sv', 'chk'): 'Helsinki-NLP/opus-mt-sv-chk', ('sv', 'crs'): 'Helsinki-NLP/opus-mt-sv-crs', ('sv', 'cs'): 'Helsinki-NLP/opus-mt-sv-cs', ('sv', 'ee'): 'Helsinki-NLP/opus-mt-sv-ee', ('sv', 'efi'): 'Helsinki-NLP/opus-mt-sv-efi', ('sv', 'el'): 'Helsinki-NLP/opus-mt-sv-el', ('sv', 'en'): 'Helsinki-NLP/opus-mt-sv-en', ('sv', 'eo'): 'Helsinki-NLP/opus-mt-sv-eo', ('sv', 'es'): 'Helsinki-NLP/opus-mt-sv-es', ('sv', 'et'): 'Helsinki-NLP/opus-mt-sv-et', ('sv', 'fi'): 'Helsinki-NLP/opus-mt-sv-fi', ('sv', 'fj'): 'Helsinki-NLP/opus-mt-sv-fj', ('sv', 'fr'): 'Helsinki-NLP/opus-mt-sv-fr', ('sv', 'gaa'): 'Helsinki-NLP/opus-mt-sv-gaa', ('sv', 'gil'): 'Helsinki-NLP/opus-mt-sv-gil', ('sv', 'guw'): 'Helsinki-NLP/opus-mt-sv-guw', ('sv', 'ha'): 'Helsinki-NLP/opus-mt-sv-ha', ('sv', 'he'): 'Helsinki-NLP/opus-mt-sv-he', ('sv', 'hil'): 'Helsinki-NLP/opus-mt-sv-hil', ('sv', 'ho'): 'Helsinki-NLP/opus-mt-sv-ho', ('sv', 'hr'): 'Helsinki-NLP/opus-mt-sv-hr', ('sv', 'ht'): 'Helsinki-NLP/opus-mt-sv-ht', ('sv', 'hu'): 'Helsinki-NLP/opus-mt-sv-hu', ('sv', 'id'): 'Helsinki-NLP/opus-mt-sv-id', ('sv', 'ig'): 'Helsinki-NLP/opus-mt-sv-ig', ('sv', 'ilo'): 'Helsinki-NLP/opus-mt-sv-ilo', ('sv', 'is'): 'Helsinki-NLP/opus-mt-sv-is', ('sv', 'iso'): 'Helsinki-NLP/opus-mt-sv-iso', ('sv', 'kg'): 'Helsinki-NLP/opus-mt-sv-kg', ('sv', 'kqn'): 'Helsinki-NLP/opus-mt-sv-kqn', ('sv', 'kwy'): 'Helsinki-NLP/opus-mt-sv-kwy', ('sv', 'lg'): 'Helsinki-NLP/opus-mt-sv-lg', ('sv', 'ln'): 'Helsinki-NLP/opus-mt-sv-ln', ('sv', 'lu'): 'Helsinki-NLP/opus-mt-sv-lu', ('sv', 'lua'): 'Helsinki-NLP/opus-mt-sv-lua', ('sv', 'lue'): 'Helsinki-NLP/opus-mt-sv-lue', ('sv', 'lus'): 'Helsinki-NLP/opus-mt-sv-lus', ('sv', 'lv'): 'Helsinki-NLP/opus-mt-sv-lv', ('sv', 'mfe'): 'Helsinki-NLP/opus-mt-sv-mfe', ('sv', 'mh'): 'Helsinki-NLP/opus-mt-sv-mh', ('sv', 'mos'): 'Helsinki-NLP/opus-mt-sv-mos', ('sv', 'mt'): 'Helsinki-NLP/opus-mt-sv-mt', ('sv', 'niu'): 'Helsinki-NLP/opus-mt-sv-niu', ('sv', 'nl'): 'Helsinki-NLP/opus-mt-sv-nl', ('sv', 'no'): 'Helsinki-NLP/opus-mt-sv-no', ('sv', 'nso'): 'Helsinki-NLP/opus-mt-sv-nso', ('sv', 'ny'): 'Helsinki-NLP/opus-mt-sv-ny', ('sv', 'pag'): 'Helsinki-NLP/opus-mt-sv-pag', ('sv', 'pap'): 'Helsinki-NLP/opus-mt-sv-pap', ('sv', 'pis'): 'Helsinki-NLP/opus-mt-sv-pis', ('sv', 'pon'): 'Helsinki-NLP/opus-mt-sv-pon', ('sv', 'rnd'): 'Helsinki-NLP/opus-mt-sv-rnd', ('sv', 'ro'): 'Helsinki-NLP/opus-mt-sv-ro', ('sv', 'ru'): 'Helsinki-NLP/opus-mt-sv-ru', ('sv', 'run'): 'Helsinki-NLP/opus-mt-sv-run', ('sv', 'rw'): 'Helsinki-NLP/opus-mt-sv-rw', ('sv', 'sg'): 'Helsinki-NLP/opus-mt-sv-sg', ('sv', 'sk'): 'Helsinki-NLP/opus-mt-sv-sk', ('sv', 'sl'): 'Helsinki-NLP/opus-mt-sv-sl', ('sv', 'sm'): 'Helsinki-NLP/opus-mt-sv-sm', ('sv', 'sn'): 'Helsinki-NLP/opus-mt-sv-sn', ('sv', 'sq'): 'Helsinki-NLP/opus-mt-sv-sq', ('sv', 'srn'): 'Helsinki-NLP/opus-mt-sv-srn', ('sv', 'st'): 'Helsinki-NLP/opus-mt-sv-st', ('sv', 'sv'): 'Helsinki-NLP/opus-mt-sv-sv', ('sv', 'swc'): 'Helsinki-NLP/opus-mt-sv-swc', ('sv', 'th'): 'Helsinki-NLP/opus-mt-sv-th', ('sv', 'tiv'): 'Helsinki-NLP/opus-mt-sv-tiv', ('sv', 'tll'): 'Helsinki-NLP/opus-mt-sv-tll', ('sv', 'tn'): 'Helsinki-NLP/opus-mt-sv-tn', ('sv', 'to'): 'Helsinki-NLP/opus-mt-sv-to', ('sv', 'toi'): 'Helsinki-NLP/opus-mt-sv-toi', ('sv', 'tpi'): 'Helsinki-NLP/opus-mt-sv-tpi', ('sv', 'ts'): 'Helsinki-NLP/opus-mt-sv-ts', ('sv', 'tum'): 'Helsinki-NLP/opus-mt-sv-tum', ('sv', 'tvl'): 'Helsinki-NLP/opus-mt-sv-tvl', ('sv', 'tw'): 'Helsinki-NLP/opus-mt-sv-tw', ('sv', 'ty'): 'Helsinki-NLP/opus-mt-sv-ty', ('sv', 'uk'): 'Helsinki-NLP/opus-mt-sv-uk', ('sv', 'umb'): 'Helsinki-NLP/opus-mt-sv-umb', ('sv', 've'): 'Helsinki-NLP/opus-mt-sv-ve', ('sv', 'war'): 'Helsinki-NLP/opus-mt-sv-war', ('sv', 'wls'): 'Helsinki-NLP/opus-mt-sv-wls', ('sv', 'xh'): 'Helsinki-NLP/opus-mt-sv-xh', ('sv', 'yap'): 'Helsinki-NLP/opus-mt-sv-yap', ('sv', 'yo'): 'Helsinki-NLP/opus-mt-sv-yo', ('sv', 'zne'): 'Helsinki-NLP/opus-mt-sv-zne', ('swc', 'en'): 'Helsinki-NLP/opus-mt-swc-en', ('swc', 'es'): 'Helsinki-NLP/opus-mt-swc-es', ('swc', 'fi'): 'Helsinki-NLP/opus-mt-swc-fi', ('swc', 'fr'): 'Helsinki-NLP/opus-mt-swc-fr', ('swc', 'sv'): 'Helsinki-NLP/opus-mt-swc-sv', ('taw', 'en'): 'Helsinki-NLP/opus-mt-taw-en', ('th', 'en'): 'Helsinki-NLP/opus-mt-th-en', ('th', 'fr'): 'Helsinki-NLP/opus-mt-th-fr', ('ti', 'en'): 'Helsinki-NLP/opus-mt-ti-en', ('tiv', 'en'): 'Helsinki-NLP/opus-mt-tiv-en', ('tiv', 'fr'): 'Helsinki-NLP/opus-mt-tiv-fr', ('tiv', 'sv'): 'Helsinki-NLP/opus-mt-tiv-sv', ('tl', 'de'): 'Helsinki-NLP/opus-mt-tl-de', ('tl', 'en'): 'Helsinki-NLP/opus-mt-tl-en', ('tl', 'es'): 'Helsinki-NLP/opus-mt-tl-es', ('tl', 'pt'): 'Helsinki-NLP/opus-mt-tl-pt', ('tll', 'en'): 'Helsinki-NLP/opus-mt-tll-en', ('tll', 'es'): 'Helsinki-NLP/opus-mt-tll-es', ('tll', 'fi'): 'Helsinki-NLP/opus-mt-tll-fi', ('tll', 'fr'): 'Helsinki-NLP/opus-mt-tll-fr', ('tll', 'sv'): 'Helsinki-NLP/opus-mt-tll-sv', ('tn', 'en'): 'Helsinki-NLP/opus-mt-tn-en', ('tn', 'es'): 'Helsinki-NLP/opus-mt-tn-es', ('tn', 'fr'): 'Helsinki-NLP/opus-mt-tn-fr', ('tn', 'sv'): 'Helsinki-NLP/opus-mt-tn-sv', ('to', 'en'): 'Helsinki-NLP/opus-mt-to-en', ('to', 'es'): 'Helsinki-NLP/opus-mt-to-es', ('to', 'fr'): 'Helsinki-NLP/opus-mt-to-fr', ('to', 'sv'): 'Helsinki-NLP/opus-mt-to-sv', ('toi', 'en'): 'Helsinki-NLP/opus-mt-toi-en', ('toi', 'es'): 'Helsinki-NLP/opus-mt-toi-es', ('toi', 'fi'): 'Helsinki-NLP/opus-mt-toi-fi', ('toi', 'fr'): 'Helsinki-NLP/opus-mt-toi-fr', ('toi', 'sv'): 'Helsinki-NLP/opus-mt-toi-sv', ('tpi', 'en'): 'Helsinki-NLP/opus-mt-tpi-en', ('tpi', 'sv'): 'Helsinki-NLP/opus-mt-tpi-sv', ('tr', 'ar'): 'Helsinki-NLP/opus-mt-tr-ar', ('tr', 'az'): 'Helsinki-NLP/opus-mt-tr-az', ('tr', 'en'): 'Helsinki-NLP/opus-mt-tr-en', ('tr', 'eo'): 'Helsinki-NLP/opus-mt-tr-eo', ('tr', 'es'): 'Helsinki-NLP/opus-mt-tr-es', ('tr', 'fr'): 'Helsinki-NLP/opus-mt-tr-fr', ('tr', 'lt'): 'Helsinki-NLP/opus-mt-tr-lt', ('tr', 'sv'): 'Helsinki-NLP/opus-mt-tr-sv', ('tr', 'uk'): 'Helsinki-NLP/opus-mt-tr-uk', ('trk', 'en'): 'Helsinki-NLP/opus-mt-trk-en', ('ts', 'en'): 'Helsinki-NLP/opus-mt-ts-en', ('ts', 'es'): 'Helsinki-NLP/opus-mt-ts-es', ('ts', 'fi'): 'Helsinki-NLP/opus-mt-ts-fi', ('ts', 'fr'): 'Helsinki-NLP/opus-mt-ts-fr', ('ts', 'sv'): 'Helsinki-NLP/opus-mt-ts-sv', ('tum', 'en'): 'Helsinki-NLP/opus-mt-tum-en', ('tum', 'es'): 'Helsinki-NLP/opus-mt-tum-es', ('tum', 'fr'): 'Helsinki-NLP/opus-mt-tum-fr', ('tum', 'sv'): 'Helsinki-NLP/opus-mt-tum-sv', ('tvl', 'en'): 'Helsinki-NLP/opus-mt-tvl-en', ('tvl', 'es'): 'Helsinki-NLP/opus-mt-tvl-es', ('tvl', 'fi'): 'Helsinki-NLP/opus-mt-tvl-fi', ('tvl', 'fr'): 'Helsinki-NLP/opus-mt-tvl-fr', ('tvl', 'sv'): 'Helsinki-NLP/opus-mt-tvl-sv', ('tw', 'es'): 'Helsinki-NLP/opus-mt-tw-es', ('tw', 'fi'): 'Helsinki-NLP/opus-mt-tw-fi', ('tw', 'fr'): 'Helsinki-NLP/opus-mt-tw-fr', ('tw', 'sv'): 'Helsinki-NLP/opus-mt-tw-sv', ('ty', 'es'): 'Helsinki-NLP/opus-mt-ty-es', ('ty', 'fi'): 'Helsinki-NLP/opus-mt-ty-fi', ('ty', 'fr'): 'Helsinki-NLP/opus-mt-ty-fr', ('ty', 'sv'): 'Helsinki-NLP/opus-mt-ty-sv', ('tzo', 'es'): 'Helsinki-NLP/opus-mt-tzo-es', ('uk', 'bg'): 'Helsinki-NLP/opus-mt-uk-bg', ('uk', 'ca'): 'Helsinki-NLP/opus-mt-uk-ca', ('uk', 'cs'): 'Helsinki-NLP/opus-mt-uk-cs', ('uk', 'de'): 'Helsinki-NLP/opus-mt-uk-de', ('uk', 'en'): 'Helsinki-NLP/opus-mt-uk-en', ('uk', 'es'): 'Helsinki-NLP/opus-mt-uk-es', ('uk', 'fi'): 'Helsinki-NLP/opus-mt-uk-fi', ('uk', 'fr'): 'Helsinki-NLP/opus-mt-uk-fr', ('uk', 'he'): 'Helsinki-NLP/opus-mt-uk-he', ('uk', 'hu'): 'Helsinki-NLP/opus-mt-uk-hu', ('uk', 'it'): 'Helsinki-NLP/opus-mt-uk-it', ('uk', 'nl'): 'Helsinki-NLP/opus-mt-uk-nl', ('uk', 'no'): 'Helsinki-NLP/opus-mt-uk-no', ('uk', 'pl'): 'Helsinki-NLP/opus-mt-uk-pl', ('uk', 'pt'): 'Helsinki-NLP/opus-mt-uk-pt', ('uk', 'ru'): 'Helsinki-NLP/opus-mt-uk-ru', ('uk', 'sh'): 'Helsinki-NLP/opus-mt-uk-sh', ('uk', 'sl'): 'Helsinki-NLP/opus-mt-uk-sl', ('uk', 'sv'): 'Helsinki-NLP/opus-mt-uk-sv', ('uk', 'tr'): 'Helsinki-NLP/opus-mt-uk-tr', ('umb', 'en'): 'Helsinki-NLP/opus-mt-umb-en', ('ur', 'en'): 'Helsinki-NLP/opus-mt-ur-en', ('urj', 'en'): 'Helsinki-NLP/opus-mt-urj-en', ('urj', 'urj'): 'Helsinki-NLP/opus-mt-urj-urj', ('ve', 'en'): 'Helsinki-NLP/opus-mt-ve-en', ('ve', 'es'): 'Helsinki-NLP/opus-mt-ve-es', ('vi', 'de'): 'Helsinki-NLP/opus-mt-vi-de', ('vi', 'en'): 'Helsinki-NLP/opus-mt-vi-en', ('vi', 'eo'): 'Helsinki-NLP/opus-mt-vi-eo', ('vi', 'es'): 'Helsinki-NLP/opus-mt-vi-es', ('vi', 'fr'): 'Helsinki-NLP/opus-mt-vi-fr', ('vi', 'it'): 'Helsinki-NLP/opus-mt-vi-it', ('vi', 'ru'): 'Helsinki-NLP/opus-mt-vi-ru', ('vsl', 'es'): 'Helsinki-NLP/opus-mt-vsl-es', ('wa', 'en'): 'Helsinki-NLP/opus-mt-wa-en', ('wal', 'en'): 'Helsinki-NLP/opus-mt-wal-en', ('war', 'en'): 'Helsinki-NLP/opus-mt-war-en', ('war', 'es'): 'Helsinki-NLP/opus-mt-war-es', ('war', 'fi'): 'Helsinki-NLP/opus-mt-war-fi', ('war', 'fr'): 'Helsinki-NLP/opus-mt-war-fr', ('war', 'sv'): 'Helsinki-NLP/opus-mt-war-sv', ('wls', 'en'): 'Helsinki-NLP/opus-mt-wls-en', ('wls', 'fr'): 'Helsinki-NLP/opus-mt-wls-fr', ('wls', 'sv'): 'Helsinki-NLP/opus-mt-wls-sv', ('xh', 'en'): 'Helsinki-NLP/opus-mt-xh-en', ('xh', 'es'): 'Helsinki-NLP/opus-mt-xh-es', ('xh', 'fr'): 'Helsinki-NLP/opus-mt-xh-fr', ('xh', 'sv'): 'Helsinki-NLP/opus-mt-xh-sv', ('yap', 'en'): 'Helsinki-NLP/opus-mt-yap-en', ('yap', 'fr'): 'Helsinki-NLP/opus-mt-yap-fr', ('yap', 'sv'): 'Helsinki-NLP/opus-mt-yap-sv', ('yo', 'en'): 'Helsinki-NLP/opus-mt-yo-en', ('yo', 'es'): 'Helsinki-NLP/opus-mt-yo-es', ('yo', 'fi'): 'Helsinki-NLP/opus-mt-yo-fi', ('yo', 'fr'): 'Helsinki-NLP/opus-mt-yo-fr', ('yo', 'sv'): 'Helsinki-NLP/opus-mt-yo-sv', ('zai', 'es'): 'Helsinki-NLP/opus-mt-zai-es', ('zh', 'bg'): 'Helsinki-NLP/opus-mt-zh-bg', ('zh', 'de'): 'Helsinki-NLP/opus-mt-zh-de', ('zh', 'en'): 'Helsinki-NLP/opus-mt-zh-en', ('zh', 'fi'): 'Helsinki-NLP/opus-mt-zh-fi', ('zh', 'he'): 'Helsinki-NLP/opus-mt-zh-he', ('zh', 'it'): 'Helsinki-NLP/opus-mt-zh-it', ('zh', 'ms'): 'Helsinki-NLP/opus-mt-zh-ms', ('zh', 'nl'): 'Helsinki-NLP/opus-mt-zh-nl', ('zh', 'sv'): 'Helsinki-NLP/opus-mt-zh-sv', ('zh', 'uk'): 'Helsinki-NLP/opus-mt-zh-uk', ('zh', 'vi'): 'Helsinki-NLP/opus-mt-zh-vi', ('zle', 'en'): 'Helsinki-NLP/opus-mt-zle-en', ('zle', 'zle'): 'Helsinki-NLP/opus-mt-zle-zle', ('zls', 'en'): 'Helsinki-NLP/opus-mt-zls-en', ('zls', 'zls'): 'Helsinki-NLP/opus-mt-zls-zls', ('zlw', 'en'): 'Helsinki-NLP/opus-mt-zlw-en', ('zlw', 'fiu'): 'Helsinki-NLP/opus-mt-zlw-fiu', ('zlw', 'zlw'): 'Helsinki-NLP/opus-mt-zlw-zlw', ('zne', 'es'): 'Helsinki-NLP/opus-mt-zne-es', ('zne', 'fi'): 'Helsinki-NLP/opus-mt-zne-fi', ('zne', 'fr'): 'Helsinki-NLP/opus-mt-zne-fr', ('zne', 'sv'): 'Helsinki-NLP/opus-mt-zne-sv'}

  langs = {
        "af": "Afrikaans",
        "als": "Tosk Albanian",
        "am": "Amharic",
        "an": "Aragonese",
        "ar": "Arabic",
        "arz": "Egyptian Arabic",
        "ast": "Asturian",
        "as": "Assamese",
        "av": "Avaric",
        "azb": "South Azerbaijani",
        "az": "Azerbaijani",
        "bar": "Bavarian",
        "ba": "Bashkir",
        "bcl": "Central Bikol",
        "be": "Belarusian",
        "bg": "Bulgarian",
        "bh": "Bihari",
        "bn": "Bengali",
        "bo": "Tibetan",
        "bpy": "Bishnupriya",
        "br": "Breton",
        "bs": "Bosnian",
        "bxr": "Russia Buriat",
        "ca": "Catalan",
        "cbk": "Chavacano",
        "ceb": "Cebuano",
        "ce": "Chechen",
        "ckb": "Central Kurdish",
        "cs": "Czech",
        "cv": "Chuvash",
        "cy": "Welsh",
        "da": "Danish",
        "de": "German",
        "diq": "Dimli",
        "dsb": "Lower Sorbian",
        "dv": "Dhivehi",
        "el": "Modern Greek",
        "eml": "Emilian-Romagnol",
        "en": "English",
        "eo": "Esperanto",
        "es": "Spanish",
        "et": "Estonian",
        "eu": "Basque",
        "fa": "Persian",
        "fi": "Finnish",
        "frr": "Northern Frisian",
        "fr": "French",
        "fy": "Western Frisian",
        "ga": "Irish",
        "gd": "Scottish Gaelic",
        "gl": "Galician",
        "gn": "Guarani",
        "gom": "Goan Konkani",
        "gu": "Gujarati",
        "he": "Hebrew",
        "hi": "Hindi",
        "hr": "Croatian",
        "hsb": "Upper Sorbian",
        "ht": "Haitian",
        "hu": "Hungarian",
        "hy": "Armenian",
        "ia": "Interlingua",
        "id": "Indonesian",
        "ie": "Interlingue",
        "ilo": "Iloko",
        "io": "Ido",
        "is": "Icelandic",
        "it": "Italian",
        "ja": "Japanese",
        "jbo": "Lojban",
        "jv": "Javanese",
        "ka": "Georgian",
        "kk": "Kazakh",
        "km": "Central Khmer",
        "kn": "Kannada",
        "ko": "Korean",
        "krc": "Karachay-Balkar",
        "ku": "Kurdish",
        "kv": "Komi",
        "kw": "Cornish",
        "ky": "Kirghiz",
        "la": "Latin",
        "lb": "Luxembourgish",
        "lez": "Lezghian",
        "li": "Limburgan",
        "lmo": "Lombard",
        "lo": "Lao",
        "lrc": "Northern Luri",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "mai": "Maithili",
        "mg": "Malagasy",
        "mhr": "Eastern Mari",
        "min": "Minangkabau",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mn": "Mongolian",
        "mrj": "Western Mari",
        "mr": "Marathi",
        "ms": "Malay",
        "mt": "Maltese",
        "mwl": "Mirandese",
        "my": "Burmese",
        "myv": "Erzya",
        "mzn": "Mazanderani",
        "nah": "Nahuatl languages",
        "nap": "Neapolitan",
        "nds": "Low German",
        "ne": "Nepali",
        "new": "Newari",
        "nl": "Dutch",
        "nn": "Norwegian Nynorsk",
        "no": "Norwegian",
        "oc": "Occitan",
        "or": "Oriya",
        "os": "Ossetian",
        "pam": "Pampanga",
        "pa": "Panjabi",
        "pl": "Polish",
        "pms": "Piemontese",
        "pnb": "Western Panjabi",
        "ps": "Pushto",
        "pt": "Portuguese",
        "qu": "Quechua",
        "rm": "Romansh",
        "ro": "Romanian",
        "ru": "Russian",
        "sah": "Yakut",
        "sa": "Sanskrit",
        "scn": "Sicilian",
        "sd": "Sindhi",
        "sh": "Serbo-Croatian",
        "si": "Sinhala",
        "sk": "Slovak",
        "sl": "Slovenian",
        "so": "Somali",
        "sq": "Albanian",
        "sr": "Serbian",
        "su": "Sundanese",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "te": "Telugu",
        "tg": "Tajik",
        "th": "Thai",
        "tk": "Turkmen",
        "tl": "Tagalog",
        "tr": "Turkish",
        "tt": "Tatar",
        "tyv": "Tuvinian",
        "ug": "Uighur",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "uz": "Uzbek",
        "vec": "Venetian",
        "vi": "Vietnamese",
        "vo": "Volapük",
        "war": "Waray",
        "wa": "Walloon",
        "wuu": "Wu Chinese",
        "xal": "Kalmyk",
        "xmf": "Mingrelian",
        "yi": "Yiddish",
        "yo": "Yoruba",
        "yue": "Yue Chinese",
        "zh": "Chinese",
    }
  
  bantu_surnames = ["Dlamini", "Gumede", "Hadebe", "Ilunga", "Kamau", "Khoza", "Lubega", "M'Bala", "Mabaso", "Mabika", "Mabizela", "Mabunda", "Mabuza", "Macharia", "Madima", "Madondo", "Mahlangu", "Maidza", "Makhanya", "Malewezi", "Mamba", "Mandanda", "Mandlate", "Mangwana", "Manjate", "Maponyane", "Mapunda", "Maraire", "Masango", "Maseko", "Masemola", "Masengo", "Mashabane", "Masire", "Masondo", "Masuku", "Mataka", "Matovu", "Mbala", "Mbatha", "Mbugua", "Mchunu", "Mkhize", "Mofokeng", "Mokonyane", "Mutombo", "Ncube", "Ndagire", "Ndhlovu", "Ndikumana", "Ndiritu", "Ndlovu", "Ndzinisa", "Ngcobo", "Nkomo", "Nkosi", "Nkurunziza", "Radebe", "Tshabalala", "Tshivhumbe", "Vila"]
  
  def ___init__(self, taget_lang='en'):
    self.stopwords_en = set(stopwords.words('english'))
    self.stopwords_target_lang = {} if target_lang not in self.langs else set(stopwords.words(self.langs[target_lang].lower()))
    self.faker = {
      'es': Faker('es_ES'),
      'en': Faker('en_US'),
      'ar': Faker('ar_AA'),
      'pt': Faker('pt_PT'),
      'fr': Faker('fr_FR'),
      'hi': Faker('hi_IN'),
      'zh': Faker('zh_CN'),
    }
    # swahilit, etc.
    self.faker['en'].add_provider(internet)
    self.faker[target-lang].add_provider(internet)
    self.titles_recognizer = PatternRecognizer(supported_entity="TITLE",
                                          deny_list=["Mr.","Mrs.","Miss", "Dr."])

    self.pronoun_recognizer = PatternRecognizer(supported_entity="PRONOUN",
                                          deny_list=["he", "He", "his", "His", "she", "She", "hers" "Hers"])

    self.pronoun_swap = {'he': 'she', 'He': 'She', 'his': 'her', 'His': 'Her', \
                    'she': 'he', 'She': 'He', 'her': 'his', 'hers': 'his', 'Her': 'His', 'Hers': 'His', }

    self.title_swap = {"Mr.": "Mrs.", "Mrs.": "Mr.", "Miss": "Mr.", "Dr.": "Dr."}
    self.analyzer = AnalyzerEngine()
    self.anonymizer = AnonymizerEngine()
    self.analyzer.registry.add_recognizer(self.titles_recognizer)
    self.analyzer.registry.add_recognizer(self.pronoun_recognizer)            
    self.translate_to_en_model = self.translate_to_inter_model = self.translate_from_en_model = self.translate_from_inter_model = None
    self.translate_to_en_tokenizer = self.translate_to_inter_tokenizer = self.translate_from_en_tokenizer = translate_from_inter_tokenizer = None
    if taget_lang != 'en':
      ret = self.find_en_tran_path(target_lang)
      if ret['to_en'] and len(ret['to_en']) == 1:
        self.translate_to_en_model = AutoModel.from_pretrained(ret['to_en'][0])
        self.translate_to_en_model = torch.quantization.quantize_dynamic(self.translate_to_en, {torch.nn.Linear}, dtype=torch.qint8)
        self.translate_to_en_tokenizer  = AutoTokenizer.from_pretrained(ret['to_en'][0])
      elif ret['to_en'] and len(ret['to_en']) == 2:
        self.translate_to_inter_model = AutoModel.from_pretrained(ret['to_en'][0])
        self.translate_to_inter_model = torch.quantization.quantize_dynamic(self.translate_to_inter_model, {torch.nn.Linear}, dtype=torch.qint8)
        self.translate_to_en_model = AutoModel.from_pretrained(ret['to_en'][1])
        self.translate_to_en_model = torch.quantization.quantize_dynamic(self.translate_to_en_model, {torch.nn.Linear}, dtype=torch.qint8)
        self.translate_to_inter_tokenizer  = AutoTokenizer.from_pretrained(ret['to_en'][0])
        self.translate_to_en_tokenizer  = AutoTokenizer.from_pretrained(ret['to_en'][1])
      if ret['from_en'] and len(ret['from_en']) == 1:
        self.translate_from_en_model = AutoModel.from_pretrained(ret['from_en'][0])
        self.translate_from_en_model = torch.quantization.quantize_dynamic(self.translate_from_en_model, {torch.nn.Linear}, dtype=torch.qint8)
        self.translate_from_en_tokenizer  = AutoTokenizer.from_pretrained(ret['from_en'][0])
      elif ret['from_en'] and len(ret['from_en']) == 2:
        self.translate_from_inter_model = AutoModel.from_pretrained(ret['to_en'][1])
        self.translate_from_inter_model = torch.quantization.quantize_dynamic(self.translate_from_inter_model, {torch.nn.Linear}, dtype=torch.qint8)
        self.translate_from_en_model = AutoModel.from_pretrained(ret['to_en'][0])
        self.translate_from_en_model = torch.quantization.quantize_dynamic(self.translate_from_en_model, {torch.nn.Linear}, dtype=torch.qint8)
        self.translate_from_inter_tokenizer  = AutoTokenizer.from_pretrained(ret['to_en'][1])
        self.translate_from_en_tokenizer  = AutoTokenizer.from_pretrained(ret['to_en'][0])

  def find_en_tran_path(self, lang1):
    lang2 = 'en'
    ret = {}
    if (lang1, lang2) in self.mariam_mt:
      ret['to_en'] = [self.mariam_mt[(lang1, lang2)]]
    else:
      found = False
      top_to_from = [a[1] for a in to_from if a[0]==lang1]
      if top_to_from:
        random.shuffle(top_to_from)
        for inter in top_to_from:
          if (lang1, inter) in self.mariam_mt and (inter, lang2) in self.mariam_mt:
            ret['to_en'] = [self.mariam_mt[(lang1, inter)], self.mariam_mt[(inter, lang2)]]
            found=True
            break
      if not found:
        ret['to_en'] = []
    if (lang2, lang1) in self.mariam_mt:
      ret['from_en']=[self.mariam_mt[(lang2, lang1)]]
    else:
      found = False
      top_to_from = [a[0] for a in to_from if a[1]==lang1]
      if top_to_from:
        random.shuffle(top_to_from)
        for inter in top_to_from:
          if (lang2, inter) in self.mariam_mt and (inter, lang1) in self.mariam_mt:
            ret['from_en'] = [self.mariam_mt[(lang2, inter)], self.mariam_mt[(inter, lang1)]]
            found=True
            break
      if not found:
        ret['from_en'] = []
    return ret

  # I find that langid isn't always as good as just figuring out if a sentence has stopwords in the lang to determine if it's the lang. 
  # first do stopword check, and then call langid.classify
  # Default is "en"
  def langid_ext(self, s, exepected_lang=None, en_lang_cutoff=0.1, target_lang_cutoff=0.1):
    lang = 'en'
    sArr = [s2.strip("' 0123456789¯_§½¼¾×|†—~”—±′–’°−{}[]·-\'?,./<>!@#^&*()+-‑=:;`→¶'") for s2 in s.split()]
    if len([s2 for s2 in sArr if s2 in self.stopwords_en])/len(sArr) >= en_lang_cutoff:
      lang = 'en'
    elif len([s2 for s2 in sArr if s2 in self.stopwords_target_lang])/len(sArr) >= target_lang_cutoff:
      lang = self.target_lang
    else:
      try:
        lang =  langid.classify(s)[0]
      except:
        lang = ""
      if lang !='en':
        ln = len(s)
        if ln < 20:
          return lang
        try:
          lang =  langid.classify(s[:int(ln/2)])[0]
          lang2 =  langid.classify(s[int(ln/2):])[0]
          if lang == 'en' or lang2 == 'en':
            lang = 'en'
        except:
          lang = 'en'
    
    if target_lang == 'fa' and expected_lang == 'ar':
      target_lang = 'ar'
    return lang

  #TODO: use Faker's lists as simple gazetter as fallback for when persido/spacy fails. For example, do firstname, lastname from ar with:
  #ArAAProvider.first_name_males
  #ArAAProvider.first_name_females
  #ArAAProvider.lastnames
  #person_gazetteer_recognizer = PatternRecognizer(supported_entity="PERSON2", ...

  #TODO: detect all the other types of PII covered by presideio, and use male/female/non-binary names provided by faker based on detected gender.
  #check if faker.name() does proportionate male/female or just picks at random in the firstname, lastname list.
  #do basic gender anaphoric matching to switch gender of name based on title and/or pronoun. Mr. male_firstname lastname -> Ms. female_firstname lastname 
  @staticmethod
  def anonymize_faker_lambda(analyzer_results, text_to_anonymize, target_lang='en', already_replaced = {}):      
      anonymized_results = anonymizer.anonymize(
          text=text_to_anonymize,
          analyzer_results=analyzer_results,    
          operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<unk>"}), \
                    "PERSON": OperatorConfig("custom", {"lambda": lambda x: faker[target_lang].name()}),\
                    "TITLE": OperatorConfig("custom", {"lambda": lambda x: title_swap.get(x, "Mrs.")}),\
                    "PRONOUN": OperatorConfig("custom", {"lambda": lambda x: pronoun_swap.get(x, "they")}),\
                    "PHONE_NUMBER": OperatorConfig("custom", {"lambda": lambda x: faker[target_lang].phone_number()}),\
                    "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": lambda x: faker[target_lang].safe_email()})}
      )

      return anonymized_results


# we should add a salt to make it more secure (do we do a diff salt per dataset?, dataset shard?)
# what we mainly want to avoid is if there was a seurity incident and the dataset of names we might have gathered
# from a webcrawled index of licensed dataset is exposed. 
  @staticmethod
  def encrypt(s):
    return (base64.b64encode(hashlib.sha512((s.strip().lower()).encode()).digest()))

  @staticmethod
  def add_label_mask(x, label_to_mask):
      #print (type(x), x) 
      id = len(label_to_mask)+1
      if x not in label_to_mask:
        label_to_mask[x] = "<"+str(id)+">"
      return label_to_mask[x]

  @staticmethod
  def template_lambda(analyzer_results, text_to_anonymize, target_lang='en', already_replaced = {}, label_to_mask={}):      
      anonymized_results = anonymizer.anonymize(
          text=text_to_anonymize,
          analyzer_results=analyzer_results,    
          operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<unk>"}), \
                    "PERSON": OperatorConfig("custom", {"lambda": lambda x: add_label_mask("PERSON", label_to_mask)}),\
                    "TITLE": OperatorConfig("custom", {"lambda": lambda x: x}),\
                    "PRONOUN": OperatorConfig("custom", {"lambda": lambda x: x}),\
                    "PHONE_NUMBER": OperatorConfig("custom", {"lambda": lambda x: add_label_mask("PHONE_NUMBER", label_to_mask)}),\
                    "EMAIL_ADDRESS": OperatorConfig("custom", {"lambda": lambda x: add_label_mask("EMAIL_ADDRESS",  label_to_mask)})}
      )

      return anonymized_results

  # TODO, for masakhaner detector, skip the translation step to english, and detect the NER types in native lang, and use faker to replace with fake names, date, etc.
  #masakhaner = pipeline("ner", "Davlan/xlm-roberta-large-masakhaner")
  # detects, person, date, location, org, etc. I think it doesn't cover things like health records, etc.
  #covers these lang:
  #Amharic, Hausa, Igbo, Kinyarwanda, Luganda, Nigerian Pidgin, Swahilu, Wolof, and Yorùbá
  #example = "Emir of Kano turban Zhang wey don spend 18 years for Nigeria"
  #ner_results = masakhaner(example)
  def translate(self, s, fr, to):
    if fr == to:
      return s
    if fr not in faker or to not in faker:
      raise RuntimeError(f"Langs other than {fr} or {to} not yet implemented")
    if fr == "en":
      return self.translate_en_ar(s)[0]['translation_text']
    return self.translate_ar_en(s)[0]['translation_text']

  # we use back translate to fix grammar problems, etc. we might introduce by doing anonymization and replacement.
  @staticmethod
  def back_trans (s, fr='en', intermediate='ar', to='en'): 
    if fr != to:
      return translate(s, fr=fr, to=to)
    return translate(translate(s, fr=fr, to=intermediate), fr=itermediate, to=to)

  # splitting doesn't always work because some languages aren't space separated
  @staticmethod
  def get_aligned_text(sent1, sent2, target_lang):
    if target_lang != "zh":
      sep = " "
      sent1 = sent1.split()
      sent2 = sent2.split()
    else:
      sep = ""
    aMatch = difflib.SequenceMatcher(None,sent1, sent2)
    score0 = aMatch.ratio()
    blocks = aMatch.get_matching_blocks()
    blocks2 = []
    prevEndA = 0
    prevEndB = 0
    matchLen = 0
    nonMatchLen = 0
    for blockI in range(len(blocks)):
        if blockI > 0 or (blockI==0 and (blocks[blockI][0] != 0 or blocks[blockI][1] != 0)):
            blocks2.append([sep.join(sent1[prevEndA:blocks[blockI][0]]), sep.join(sent2[prevEndB:blocks[blockI][1]]), 0])
            nonMatchLen += max(blocks[blockI][0] - prevEndA, blocks[blockI][1] - prevEndB)
        if blocks[blockI][2] != 0:
          blocks2.append([sep.join(sent1[blocks[blockI][0]:blocks[blockI][0]+blocks[blockI][2]]), sep.join(sent2[blocks[blockI][1]:blocks[blockI][1]+blocks[blockI][2]]), 1])
          prevEndA = blocks[blockI][0]+blocks[blockI][2]
          prevEndB = blocks[blockI][1]+blocks[blockI][2]
          matchLen += blocks[blockI][2]
    score = float(matchLen+1)/float(nonMatchLen+1)
    return (blocks2, score, score0)

  # todo, do gender swapping
  @staticmethod
  def anonymize_with_faker(anonymized_text, label_to_mask, target_lang):
      for label, mask in label_to_mask.items():
        val = mask
        if label == "PERSON":
          val = faker[target_lang].name()
        elif label == "PHONE_NUMBER":
          val = faker[target_lang].phone_number()
        elif label == "PHONE_NUMBER":
          val = faker[target_lang].phone_number()
        elif label == "EMAIL_ADDRESS":
          val = faker[target_lang].safe_email()
        anonymized_text = anonymized_text.replace(mask, val)
      return anonymized_text

  @staticmethod
  def PII_process_with_back_trans_method_1():
    (blocks2, score, score0) =  get_aligned_text(text_to_anonymize, anonymized_text0, target_lang)
    print (blocks2)
    #since we know what slots we are replacing, we can align with the original sentence in the original language, and then insert the fake value for the slots. 
    mask_to_pii_encrypt = {}
    mask_to_label = dict([(b, a) for a, b in label_to_mask.items()])
    label_to_val={}
    anonymized_text_pii_test = text_to_anonymize
    for (s1, s2, matched) in blocks2:
      if "<" in s2 and ">" in s2:
        s2 = s2.split("<")[-1].split(">")[0]
        print ('matched', "<"+s2+">", '**', s1)
        mask_to_pii_encrypt["<"+s2+">"]  = encrypt(s1)
        label_to_val[mask_to_label["<"+s2+">"]] = s1
        anonymized_text_pii_test= anonymized_text_pii_test.replace(s1, "<"+s2+">")


  #NOTE: Kenlm is LGPL, but we are not modifying Kenlm. We are only using it's API, so the source code of this module will not be subject to the LGPL, and will stay as Apache.
  @staticmethod
  def kenlm(infiles, outfile, ngram_size=5, ngram_cnt_cutoff=5):
    if type(infiles) is list:
      infiles = " ".join(infiles)
    cmd = f"lmplz --discount_fallback  --skip_symbols -o {ngram_size} --prune {ngram_cnt_cutoff} --collapse_values  --arpa {outfile} < {infiles}"
    print (cmd)
    os.system(cmd)

  # todo, we will need to tokenize languages that do not use spaces such as zh to seperate words. insert spaces in between words, and word ngram??
  @staticmethod
  # note that passing all these dictionaries around could be expensive esp. if we are sending to different servers. 
  def cleanup_and_tokenize(dataset, shard_range, ngram_size, lang, fingerprint, first_parse, ngram, data_stopwords, compound_start, word2ontology, last_parse):
    if type(dataset) is str:
      with open(dataset, "rb") as f:
        f.seek(shard_range[0])
        batch = f.read(shard_range[1]-shard_range[0]).decode().split("\n")
    else:
      batch = dataset[shard_range[0]:shard_range[1]]['text']
    shard_num = int(shard_range[0])
    with open(f"{lang}_oscar_tok_{fingerprint}_{shard_num}.txt", "a+", encoding="utf8") as f:
      while batch:
        sent = batch[0]
        batch = batch[1:]
        if first_parse:
          sent = [s2.strip("،♪↓↑→←━\₨₡€¥£¢¤™®©¶§←«»⊥∀⇒⇔√­­♣️♥️♠️♦️‘’¿*’-ツ¯‿─★┌┴└┐▒∎µ•●°¦¬≥≤±≠¡×÷¨´:।`~�_“”/|!~@#$%^&*•()[]{}-_+–=<>·;…?:.,\'\"") \
              for s2 in sent.lower().replace('�', ' ').replace('�', ' ').split() if s2.strip("،♪↓↑→←━\₨₡€¥£¢¤™®©¶§←«»⊥∀⇒⇔√­­♣️♥️♠️♦️‘’¿*’-ツ¯‿─★┌┴└┐▒∎µ•●°¦¬≥≤±≠¡×÷¨´:।`~�_“”/|!~@#$%^&*•()[]{}-_+–=<>·;…?:.,\'\"")] 
        else:
          sent = sent.split()
        len_sent = len(sent)
        for i in range(len_sent-1):
          if sent[i] is None: continue
          if sent[i].split("_")[0] in compound_start:
            for j in range(ngram_size-1, -1, -1):
              if len_sent - i  > j:
                word = "_".join(sent[i:i+1+j])
                if word in ngram:
                  sent[i] = "_".join(sent[i:i+1+j])
                  for k in range(i, i+j):
                    sent[k] = None  
                  break
          if last_parse:
            if "_" in sent[i] and sent[i].split("_")[-1] in data_stopwords:
              word_arr = sent[i].split("_")
              word2 = ""
              for k in range(len(word_arr)-1, -1, -1):
                if word_arr[k] in data_stopwords:
                  word2 = " "+word_arr[k]+word2
                else:
                  word2 = "_".join(word_arr[0:k+1])+word2
                  break
              sent[i] = word2.strip("_ ")
          
          f.write(" ".join([s if s not in word2ontology else word2ontology[s] for s in sent if s])+"\n")
        
  @staticmethod
  def get_file_shard_ranges(input_file_path, batch_size, batch_per_shard, num_proc, num_shards=None):
        file_size= os.stat(input_file_path).st_size        
        if num_shards is not None:
            shard_size = int(file_size/num_shards) # if there is a remainder, we might need to add a little bit the shard_size??
        else:
          shard_size = batch_size*batch_per_shard
        shard_size = min(shard_size, int(len(file_size)/num_proc))
        with open(input_file_path, "rb") as f:
          file_segs = []
          file_pos = 0
          while file_pos < file_size:
                if file_size - file_pos <= shard_size:
                    file_segs.append((file_pos, file_size))
                    break
                f.seek(file_pos+shard_size, 0)
                seg_len = shard_size
                line = f.readline()
                if not line:
                    file_segs.append((file_pos, file_size))
                    break
                seg_len += len(line)
                if file_size-(file_pos+seg_len) < shard_size:
                    file_segs.append((file_pos, file_size))
                    break

                file_segs.append((file_pos, file_pos + seg_len))
                file_pos = f.tell()
          line = None
          return file_segs

  @staticmethod
  def get_dataset_shard_ranges(dataset, batch_size, batch_per_shard, num_proc):
      shard_size = batch_size*batch_per_shard
      shard_size = min(shard_size, int(len(dataset)/num_proc))
      shard_ranges = []
      len_dataset = len(dataset)
      for rng in range(0, len_dataset, shard_size):
        max_rng = min(rng+shard_size,len_dataset )
        shard_ranges.append((rng, max_rng))
      return shard_ranges

  @staticmethod
  def create_oscar_ontology(langs, dask_type=['dask'], ngram = {}, ontology = {}, word2ontology={}, compound_start = {}, batch_per_shard=1, ngram_size=5, ngram_discovery_times=1, ngram_cnt_cutoff=5, batch_size=50000, epoch=10, size_mb= 20000, max_num_stopwords=100, stopword_max_len = 7, words_per_cluster=10, num_proc=4):
    # TODO: create temporary cache files in the same tmp folder as the datasets
    # hook into save_datasets to save corresponding models and metadata so everything is saved in one unit
    
    with parallel_backend(*dask_type): # for multi-node, set dask_type=['distributed', 'HOST:PORT']:
      langs.sort()
      lang = "+".join(langs)
      for a_lang in langs:
        data_tok_or_file = load_dataset("oscar", f"unshuffled_deduplicated_{a_lang}")
        data_tok_or_file = data_tok_or_file['train']
        # write to a bunch of shard files.
        # delete the /root/.cache/huggingface folder to save memory.
        os.system("rm -rf /root/.cache/huggingface")
      batch_size = min(batch_size, int(len(data_tok_or_file)/num_proc))
      fingerprint = data_tok_or_file._fingerprint
      shard_ranges = get_dataset_shard_ranges(data_tok_or_file, batch_size, batch_per_shard, num_proc)
      num_segs = len(shard_ranges)
      data_stopwords={}
      for times in range(ngram_discovery_times):
        jobs = glob.glob(f"{lang}_oscar_tok_{fingerprint}_*.txt")
        for job in jobs:
          os.unlink(job)
        # todo, delete the previous data_tok_or_file file if one exists.
        Parallel(n_jobs=num_proc, verbose=1)(delayed(cleanup_and_tokenize)(data_tok_or_file, shard_range, ngram_size, lang, fingerprint, times==0, ngram, data_stopwords, compound_start, word2ontology, False) for shard_range in shard_ranges)
        jobs = glob.glob(f"{lang}_oscar_tok_{fingerprint}_*.txt")
        all_jobs_shards = " ".join(jobs)
        data_tok_or_file = f"{lang}_oscar_tok_{fingerprint}.txt"
        os.system(f"cat {all_jobs_shards} > {data_tok_or_file}")
        shard_ranges = get_file_shard_ranges(data_tok_or_file, num_segs=num_segs)
        print (shard_ranges)
        for job in jobs:
          os.unlink(job)
        outfile = f"{lang}_oscar_tok_{fingerprint}.arpa"
        kenlm(data_tok_or_file, outfile, ngram_size=ngram_size, ngram_cnt_cutoff=ngram_cnt_cutoff)
        with open(outfile, "rb") as f:
          with open (f"{lang}_oscar_unigram_{fingerprint}.tsv", "w", encoding="utf8") as o:
            is_unigram = False
            unigram={}
            for line in  f.read().decode().split("\n"): # do this incrementally if we need to. 
              line = line.strip()
              if "2-grams:" in line:
                unigram_list = list(unigram.items())
                unigram_list.sort(key=lambda a: a[1], reverse=True)
                data_stopwords = dict([item for item in unigram_list[:200] if len(item[0]) < stopword_max_len][:100])
                print ('stop words', len(data_stopwords), data_stopwords)
                unigram = None
                is_unigram = False
              elif is_unigram and line:
                try:
                  weight, word, _ = line.split()
                  if word in ('<unk>', '<s>', '</s>'):
                    continue
                except:
                  print ('**problem line**', line)
                  continue
                weight = math.exp(float(weight))
                o.write (word+"\t"+str(weight)+"\n")
                unigram[word] = weight
              elif "1-grams:" in line and times == 0: 
                # do we really want to recreate the unigram list every cycle. will the data change?
                # we will get compound words, but that's already captured by the ngram.
                is_unigram = True
              else:
                line = line.split()
                if len(line) > 3:
                  weight = float(line[0])
                  line = line[1:len(line)-1]
                  if line[0] in data_stopwords or [l in ('<unk>', '<s>', '</s>') for l in line]:
                    continue
                  ngram["_".join(line)]=  math.exp(float(weight))
                  compound_start[line[0]]=1
            unigram=None
              
        # fasttext training
        if not os.path.exists(f"{lang}_oscar_fastext_model_{fingerprint}.bin"):
          print ("training fastext model")
          model = fasttext.train_unsupervised(data_tok_or_file, epoch=epoch)
          model.save_model(f"{lang}_oscar_fastext_model_{fingerprint}.bin")
        else:
          model = fasttext.load_model(f"{lang}_oscar_fastext_model_{fingerprint}.bin")
        # now do clustering
        terms = model.get_words()
        if  os.path.exists( f'{lang}_oscar_ontology_{fingerprint}.joblib'):
            km = load( f'{lang}_oscar_ontology_{fingerprint}.joblib') 
        else:
          print ("clustering")
          true_k=int(len(terms)/words_per_cluster)
          # what is the difference between get_word_vector and the get_input_matrix and get_output_matrix.
          # does get_input_matrix correspond to the word_vectors, we don't have to do the vstack and waste memory.
          # to confirm. 
          # TODO, use memmap and dask kmeans to do parrallel distributed out-of-core clustering.
          x = model.get_input_matrix()
          #x=np.vstack([model.get_word_vector(term) for term in model.get_words()])

          # TODO: combine this embedding with other embeddings such as conceptnet/numberbatch
          #k-means with word_vectors
          #TODO: if the embeddings can't fit in memory do mmemmap, and run in parrallel with https://ml.dask.org/modules/generated/dask_ml.cluster.KMeans.html ??
          # NOTE: k-means|| used by dask and the k-means used by dask is too slow. We need to use something else. And minibatchkmeans can't handle in reasonable time many millions of words.

          # proposed to use creating N indexing words, and grouping those words and sub-word groups into shards. and sending each shard to a node to do minibatchkmeans.
          # 300K words can be clustered in around 30 mins. 
          # We can break down 1.5M words into 5 shards for example to run on 5 servers.  
          # We need a mechanism for sending these shards to each node.
          # Dask does this through tcp, 
          # But for tsting in colab, that might not work well in colab since there is no peer-to-peer communicaiton between colab instances, and at most we can have each colab 
          # instance talk to a master colab instance. We might need to just save all shards to a shared google drive. 
          # this will be easier when we can run in production on a real cluster. 

          km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                                  init_size=max(true_k*3,1000), batch_size=1000).fit(x)
          dump(km, f'{lang}_oscar_ontology_{fingerprint}.joblib') 
        
        for term, label in zip(terms, km.labels_):
          # todo, get the class label with max coverage here and replace label with class label, with the winner. 
          # decluster anything that isn't in the max class. 
          # TODO. do word2ontology mapping here.
          pass
        # now recreate the ontoloy based on word2ontology. ontology[label] = ontology.get(label, [])+[term]
        json.dumps(ontology, open(f'{lang}_oscar_ontology_{fingerprint}.json', 'w', encoding='utf8'))

      return ontology

  @staticmethod
  def annotate_PII(cls, target_lang, batch, out_file):
    pii_mixin = cls(target_lang0)
    for line in batch:
      text_to_anonymize = line.get(f'text_{target_lang}', line.get("text"), '')
      if not text_to_anonymize:
        continue
      # we do translation to english because the tools we use work in english mostly. we translate back to target language at the end.  
      if target_lang != pii_mixin.langid_ext(text_to_anonymize,):
        print (f"not in target lang {target_lang}: {text_to_anonymize}")
        continue
      if target_lang !='en':

          # langid_ext doesn't quite work. detects arabic as farsi. 
        print ('detected lang', target_lang)
        text_to_anonymize_en = translate(text_to_anonymize, fr=target_lang, to='en')
        print(text_to_anonymize_en)
      analyzer_results = analyzer.analyze(text=text_to_anonymize_en, language='en')
      print (analyzer_results)
      label_to_mask = {}
      pii_test_set = template_lambda(analyzer_results, text_to_anonymize_en, target_lang=target_lang, already_replaced={}, label_to_mask=label_to_mask) 
      print (pii_test_set.text)
      anonymized_text0 =  back_trans(pii_test_set.text, fr='en', to=target_lang)
      (blocks2, score, score0) =  get_aligned_text(text_to_anonymize, anonymized_text0, target_lang)
      print (blocks2)
      #since we know what slots we are replacing, we can align with the original sentence in the original language, and then insert the fake value for the slots. 
      mask_to_pii_encrypt = {}
      mask_to_label = dict([(b, a) for a, b in label_to_mask.items()])
      label_to_val={}
      anonymized_text_pii_test = text_to_anonymize
      for (s1, s2, matched) in blocks2:
        if "<" in s2 and ">" in s2:
          s2 = s2.split("<")[-1].split(">")[0]
          print ('matched', "<"+s2+">", '**', s1)
          mask_to_pii_encrypt["<"+s2+">"]  = encrypt(s1)
          label_to_val[mask_to_label["<"+s2+">"]] = s1
          anonymized_text_pii_test= anonymized_text_pii_test.replace(s1, "<"+s2+">")
      print ('pii test set', anonymized_text_pii_test, '**', mask_to_pii_encrypt)
      anonymized_text1 = anonymize_with_faker(text_to_anonymize, label_to_val, target_lang) # maybe we run this through back_trans using some intermediate lang??
      print ('aonymized method 1', anonymized_text1)
      label_to_val = None # make sure we delete this dict as it has the actual PII
      anonymized_text2 = anonymize_with_faker(anonymized_text0, label_to_mask, target_lang) # maybe we run this through back_trans using some intermediate lang??
      print ('aonymized method 2', anonymized_text2)
      anonymized_results = anonymize_faker_lambda(analyzer_results, text_to_anonymize_en, target_lang=target_lang, already_replaced={})
      print ('aonymized method 3',  back_trans(anonymized_results.text, fr='en', to=target_lang))
      # method 2 appears to be better than 1, and 1 better than method 3  
      # TODO, do probing using Question Gen.

if __name__ == "__main__": 
  bigscience_lang = ['sl', 'hr', 'mk', 'bg', 'pl', 'uk', 'sk', 'cs', 'be', 'ru', 'fi', 'eu', 'zh', 	'vi', 'fr', 'es', 'id', 'pt', 'ca', 'ar', 'ur', 'hi', 'bn', 'yo', 'sw', 'ng', 'rw', 'ig', 'ha', ] #en
  pass


        

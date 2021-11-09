"""Basic filtering of garbage and perplexity sampling for OSCAR v1."""

import gzip
import multiprocessing
import os
from random import sample

import fsspec
import kenlm  # pip install https://github.com/kpu/kenlm/archive/master.zip
import langid
import numpy as np
from datasets import load_dataset
from nltk.corpus import stopwords
from numpy.random import default_rng
from transformers import AutoTokenizer


class OscarSampler:
    """Based on bertin/mc4/mc4.py.
    This code does not use HF's datasets for efficiency reasons."""

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
        "nah": "Nahuatl",  # languages
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

    stopwords_cutoff = 0.1
    junk_ratio = 0.5
    stopword_check = True
    special_characters = (
        "' 0123456789¯_%$§½¼¾×|†—~\"—±′–'°−{}[]·-'?,./<>!@#^&*()+-‑=:;`→¶'€"
    )

    # TODO - add params for other languages
    params = {
        "en": {
            "stopwords_cutoff": stopwords_cutoff,
            "junk_ratio": junk_ratio,
            "stopword_check": stopword_check,
            "strip_chars": special_characters,
            "junk_chars": special_characters,
        },

        "fr": {
            "stopwords_cutoff": 0.1,
            "junk_ratio": 0.5,
            "stopword_check": True,
            "strip_chars": "' 0123456789¯_%$§½¼¾×|†—~\"—±′–'°−{}[]·-'?,./<>!@#^&*()+-‑=:;`→¶'€«»",
            "junk_chars": "' 0123456789¯_%$§½¼¾×|†—~\"—±′–'°−{}[]·-'?,./<>!@#^&*()+-‑=:;`→¶'€«»",
        }
    }

    def __init__(self, **kwargs):
        self.sampling_method = kwargs.pop("sampling_method", "random")
        self.perplexity_model = kwargs.pop("perplexity_model", None)
        self.sampling_factor = kwargs.pop("sampling_factor", None)
        self.boundaries = kwargs.pop("boundaries", None)
        if self.sampling_method:
            if self.sampling_method == "random":
                self.should_keep_doc = self._should_keep_doc_random
            else:
                # Loading 5-gram model
                # http://dl.fbaipublicfiles.com/cc_net/lm/es.arpa.bin
                print("loading model = %s", self.perplexity_model)
                if self.sampling_method == "gaussian":
                    self.should_keep_doc = self._should_keep_doc_gaussian
                else:
                    self.should_keep_doc = self._should_keep_doc_step
        self.seed = kwargs.pop("seed", None)
        self.kwargs = kwargs

    @staticmethod
    def get_oscar_urls(language, shuffled="unshuffled", deduplicated="deduplicated"):
        _BASE_DATA_URL_FORMAT_STR = "https://s3.amazonaws.com/datasets.huggingface.co/oscar/1.0/{shuffled}/{deduplicated}/{language}/"
        _BASE_CHECKSUM_FILE_NAME = "{language}_sha256.txt"
        base_data_url = _BASE_DATA_URL_FORMAT_STR.format(
            shuffled=shuffled, language=language, deduplicated=deduplicated
        )
        checksum_url = base_data_url + _BASE_CHECKSUM_FILE_NAME.format(
            language=language
        )
        with fsspec.open(checksum_url, encoding="utf-8") as f:
            data_filenames = [line.decode().split("\t")[0] for line in f if line]
        return [base_data_url + data_filename for data_filename in data_filenames]

    @staticmethod
    def _download_urls(urls):
        for url in urls:
            if not os.path.exists(url.split("/")[-1]):
                os.system(f"wget {url}")

    @staticmethod
    def check_good_sentence(
        sentence,
        stopwords,
        junk_dict,
        strip_chars,
        target_lang,
        stopwords_cutoff,
        junk_ratio,
        stopword_check,
    ):
        # basic dejunk
        sent = sentence.lower().strip()
        if not sent:
            return False
        jr = len([char for char in sent if char in junk_dict]) / len(sent)
        if jr >= junk_ratio:
            return False
        words = [word.strip(strip_chars) for word in sent.split()]
        if len(words) == 0:
            return False
        # stopword check
        if stopword_check:
            stopword_cond = (
                len([word for word in words if word in stopwords]) / len(words)
                < stopwords_cutoff
            )
            if stopword_cond:
                return False
        else:
            # langid check
            try:
                lang = langid.classify(sent)[0]
            except:
                lang = ""
            return lang == target_lang

    @staticmethod
    def filter_and_tok_cjk(
        url,
        target_lang,
        sampling_factor,
        boundaries,
        should_keep_doc,
        perplexity_model,
        seed,
        stopwords_cutoff,
        junk_ratio,
        stopword_check,
        strip_chars,
        junk_chars,
    ):
        mt5_underscore = "_"
        if seed is not None:
            rng = default_rng(seed)
        else:
            rng = default_rng()
        if perplexity_model:
            pp_model = kenlm.Model(perplexity_model)
        else:
            pp_model = None
        stopwords = set(stopwords.words(OscarSampler.langs[target_lang].lower()))
        junk_dict = {a: 1 for a in junk_chars}
        if target_lang in ("ja", "zh", "ko"):
            tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        OscarSampler._download_urls([url])
        file = url.split("/")[-1]
        with open(
            file.replace("txt.gz", "") + ".sample_filtered.txt", "w", encoding="utf8"
        ) as f:
            with gzip.open(file, "rb") as f2:
                for id_, line in enumerate(f2):
                    line = line.decode().strip()
                    if target_lang in ("ja", "zh", "ko"):
                        line = " ".join(tokenizer.tokenize(line)).replace(
                            mt5_underscore + " ", mt5_underscore
                        )
                    if OscarSampler.check_good_sentence(
                        line,
                        stopwords,
                        junk_dict,
                        strip_chars,
                        target_lang,
                        stopwords_cutoff,
                        junk_ratio,
                        stopword_check,
                    ):
                        # now do perplexity sampling
                        if should_keep_doc(
                            line,
                            rng=rng,
                            factor=sampling_factor,
                            boundaries=boundaries,
                            pp_model=pp_model,
                        ):
                            f.write(line + "\n")
        os.unlink(file)

    def sample_filter(self, target_lang, sample_shard=5):
        if target_lang in self.params:
            param = self.params[target_lang]
        else:
            param = self.params["en"]
        stopwords_cutoff = param["stopwords_cutoff"]
        junk_ratio = param["junk_ratio"]
        stopword_check = param["stopword_check"]
        strip_chars = param["strip_chars"]
        junk_chars = param["junk_chars"]
        if target_lang in self.langs:
            lst = self.get_oscar_urls(target_lang)
            if sample_shard and len(lst) > sample_shard:
                lst = sample(lst, sample_shard)
            # TODO, we should create
            processes = [
                multiprocessing.Process(
                    target=OscarSampler.filter_and_tok_cjk,
                    args=(
                        url,
                        target_lang,
                        self.sampling_factor,
                        self.boundaries,
                        self.should_keep_doc,
                        self.perplexity_model,
                        self.seed,
                        stopwords_cutoff,
                        junk_ratio,
                        stopword_check,
                        strip_chars,
                        junk_chars,
                    ),
                )
                for url in lst
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            os.system(
                f"cat {target_lang}_*.sample_filtered.txt > {target_lang}.sample_filtered.txt"
            )
            os.system(f"gzip {target_lang}.sample_filtered.txt")
            return f"{target_lang}.sample_filtered.txt.gz"  # TODO put this in a data folder.
        else:
            print(f"{target_lang} not supported")
            return ""

    @staticmethod
    def create_knlm_model(lang="pt"):
        if not os.path.exists("/content/lmplz"):
            os.system(
                "cp /content/drive/Shareddrives/BigScience/kenlm/bin/lmplz /content/"
            )
            os.system("chmod ugo+x /content/lmplz")
        file = tokenize_oscar_subset(lang, force=False)
        file2 = os.path.split(file)[-1]
        if not os.path.exists(file2) and not os.path.exists(file2.replace(".gz", "")):
            os.system(f"cp {file} ./{file2}")
        if os.path.exists(file2):
            os.system(f"gunzip ./{file2}")
        file2 = file2.replace(".gz", "")
        os.system(
            f"/content/lmplz --discount_fallback  --skip_symbols -o 5 --prune 5 --collapse_values  --arpa {lang}.arpa < ./{file2}"
        )
        os.system(f"mv {lang}.arpa /content/drive/Shareddrives/BigScience")

    @staticmethod
    def get_perplexity(doc, pp_model):
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = pp_model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return 10.0 ** (-doc_log_score / doc_length)

    @staticmethod
    def _should_keep_doc_step(doc, rng, factor=1.5e5, boundaries=None, **kwargs):
        pp_model = width = kwargs.get("pp_model")
        perplexity = OscarSampler.get_perplexity(doc, pp_model)
        if boundaries is None:
            boundaries = [536394.99320948, 662247.50212365, 919250.87225178]
        if perplexity <= boundaries[0]:
            quartile_range = boundaries[0]
        elif boundaries[0] < perplexity < boundaries[1]:
            quartile_range = boundaries[1] - boundaries[0]
        elif boundaries[1] < perplexity < boundaries[2]:
            quartile_range = boundaries[2] - boundaries[1]
        elif perplexity >= boundaries[2]:
            quartile_range = 10 * boundaries[2]
        probability = factor / quartile_range
        return rng.uniform() < probability

    @staticmethod
    def _should_keep_doc_gaussian(doc, rng, factor=0.78, boundaries=None, **kwargs):
        pp_model = width = kwargs.get("pp_model")
        width = kwargs.get("width", 9 / 2)  # width (spread) of the exponential curve
        perplexity = OscarSampler.get_perplexity(doc, pp_model)
        if boundaries is not None:
            m = boundaries[1]
        else:
            m = 662247.50212365
        exponential = np.exp((-1 / width) * ((perplexity - m) / m) ** 2)
        weighted_perplexity = factor * exponential
        return rng.uniform() < weighted_perplexity

    @staticmethod
    def _should_keep_doc_random(doc, rng, factor=None, boundaries=None, **kwargs):
        if factor is None:
            factor = 0.5
        return rng.uniform() <= factor

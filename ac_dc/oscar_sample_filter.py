"""Basic filtering of garbage and perplexity filtering for OSCAR v1.
This code does not use HF's datasets for efficiency reasons."""

import os
import fsspec
import gzip
import multiprocessing

import numpy as np
from random import sample

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords as nltk_stopwords
import fasttext
# To download the fasttext model:
# wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
import kenlm  # pip install https://github.com/kpu/kenlm/archive/master.zip

from languages_id import langs_id


class BasicFiltering:
    @staticmethod
    def lower_strip_sentence(sentence):
        sent = sentence.lower().strip()
        return sent

    @staticmethod
    def get_words_from_sentence(sentence, strip_characters):
        sent = BasicFiltering.lower_strip_sentence(sentence)
        words = [word.strip(strip_characters) for word in sent.split(" ")]
        return words

    @staticmethod
    def remove_words_with_incorrect_substrings(
        sentence,
        incorrect_word_substrings,
    ):
        words = sentence.split(" ")
        words = [
            word
            for word in words
            if all([(i_substr not in word) for i_substr in incorrect_word_substrings])
        ]
        filtered_sentence = " ".join(words)
        return filtered_sentence

    @staticmethod
    def remove_long_words(
        sentence,
        length_word_cutoff,
    ):
        words = sentence.split(" ")
        words = [word for word in words if len(word) < length_word_cutoff]
        filtered_sentence = " ".join(words)
        return filtered_sentence

    @staticmethod
    def check_empty(sentence, strip_characters):
        sent = BasicFiltering.lower_strip_sentence(sentence)
        words = BasicFiltering.get_words_from_sentence(sentence, strip_characters)
        cond = (len(sent) > 0) and (len(words) > 0)
        return cond

    @staticmethod
    def check_special_characters(
        sentence,
        special_characters,
        special_characters_cutoff,
    ):
        sent = BasicFiltering.lower_strip_sentence(sentence)
        set_special_characters = {char for char in special_characters}
        special_characters_ratio = len(
            [char for char in sent if char in set_special_characters]
        ) / len(sent)
        cond = special_characters_ratio < special_characters_cutoff
        return cond

    @staticmethod
    def check_stopwords(
        sentence,
        strip_characters,
        lang_oscar_id,
        stopwords_cutoff,
    ):
        cond = True
        nltk_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "nltk_id"
        ].iloc[0]
        if nltk_lang_id:
            words = BasicFiltering.get_words_from_sentence(sentence, strip_characters)
            stopwords = set(nltk_stopwords.words(nltk_lang_id))
            stopwords_ratio = len([word for word in words if word in stopwords]) / len(
                words
            )
            cond = stopwords_ratio < stopwords_cutoff
        return cond

    @staticmethod
    def check_badwords(
        sentence,
        strip_characters,
        lang_oscar_id,
        badwords_cutoff,
    ):
        cond = True
        badwords_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "badwords_id"
        ].iloc[0]
        if badwords_lang_id:
            words = BasicFiltering.get_words_from_sentence(sentence, strip_characters)
            f = open(f"badwords_{badwords_lang_id}.txt", "r")
            badwords = set(f.read().split("\n"))
            f.close()
            badwords_ratio = len([word for word in words if word in badwords]) / len(
                words
            )
            cond = badwords_ratio < badwords_cutoff
        return cond

    @staticmethod
    def check_lang_id(
        sentence,
        strip_characters,
        lang_oscar_id,
        path_model_fasttext,
        lang_id_cutoff,
    ):
        cond = True
        fasttext_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "fasttext_id"
        ].iloc[0]
        if fasttext_lang_id:
            words = BasicFiltering.get_words_from_sentence(sentence, strip_characters)
            sent = " ".join(words)
            model_lang_id = fasttext.load_model(path_model_fasttext)
            pred = model_lang_id.predict(sent)
            lang_pred_fasttext_id = pred[0][0].replace("__label__", "")
            score_pred = pred[1][0]
            lang_pred_oscar_id = langs_id.loc[
                langs_id["fasttext_id"] == lang_pred_fasttext_id, "oscar_id"
            ].iloc[0]
            cond = (lang_pred_oscar_id == lang_oscar_id) and (
                score_pred > lang_id_cutoff
            )
        return cond

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
            stopword_ratio = len([word for word in words if word in stopwords]) / len(
                words
            )
            stopword_cond = stopword_ratio > stopwords_cutoff
            if stopword_cond:
                return False
            return True
        else:
            # langid check
            try:
                # lang = langid.classify(sent)[0]
                pass
            except:
                lang = ""
            return lang == target_lang


class PerplexityFiltering:
    @staticmethod
    def get_perplexity(pp_model, doc):
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = pp_model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return 10.0 ** (-doc_log_score / doc_length)


class OscarFiltering:
    def __init__(
        self,
        lang_oscar_id,
        stopwords_cutoff=0.1,
        junk_ratio=0.5,
        stopword_check=True,
        strip_chars=(
            "' 0123456789¯_%$§½¼¾×|†—~\"—±′–'°−{}[]·-'?,./<>!@#^&*()+-‑=:;`→¶'"
        ),
        junk_chars=(
            "' 0123456789¯_%$§½¼¾×|†—~\"—±′–'°−{}[]·-'?,./<>!@#^&*()+-‑=:;`→¶'"
        ),
        perplexity_model=None,
    ):

        self.lang_oscar_id = lang_oscar_id

        self.stopwords_cutoff = stopwords_cutoff
        self.junk_ratio = junk_ratio
        self.stopword_check = stopword_check
        self.strip_chars = strip_chars
        self.junk_chars = junk_chars
        self.perplexity_model = perplexity_model

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

    # TODO: Finish to adapt the following function
    # make it work with the changes in the code
    @staticmethod
    def filter_and_tok_cjk(
        url,
        target_lang,
        perplexity_model,
        stopwords_cutoff,
        junk_ratio,
        stopword_check,
        strip_chars,
        junk_chars,
    ):
        if perplexity_model:
            pp_model = kenlm.Model(perplexity_model)
        else:
            pp_model = None
        stopwords = set()
        junk_dict = {a: 1 for a in junk_chars}
        OscarFiltering._download_urls([url])
        file = url.split("/")[-1]
        with open(
            file.replace(".txt.gz", "") + ".sample_filtered.txt", "w", encoding="utf8"
        ) as f:
            with gzip.open(file, "rb") as f2:
                for id_, line in enumerate(f2):
                    line = line.decode().strip()
                    if BasicFiltering.check_good_sentence(
                        line,
                        stopwords,
                        junk_dict,
                        strip_chars,
                        target_lang,
                        stopwords_cutoff,
                        junk_ratio,
                        stopword_check,
                    ):
                        f.write(line + "\n")
        os.unlink(file)

    # TODO: Finish to adapt the following function
    # make it work with the changes in the code
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
                    target=OscarFiltering.filter_and_tok_cjk,
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
                        self.sampling_method,
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


if __name__ == "__main__":
    # Download a sample dataset and run filtering pipeline
    lang = "af"
    sampler = OscarFiltering(lang_oscar_id=lang)
    url = OscarFiltering.get_oscar_urls(lang)[0]
    OscarFiltering.filter_and_tok_cjk(
        url=url,
        target_lang="af",
        stopwords_cutoff=0.1,
        junk_ratio=0.15,
        stopword_check=True,
        strip_chars=sampler.strip_chars,
        junk_chars=sampler.junk_chars,
        perplexity_model="ac_dc/af.arpa.bin",
    )

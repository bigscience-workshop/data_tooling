"""Basic filtering of garbage and perplexity filtering for OSCAR v1.
This code does not use HF's datasets for efficiency reasons."""

import os
import gzip

import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords as nltk_stopwords

import fasttext

# To download the fasttext model:
# wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

import kenlm  # pip install https://github.com/kpu/kenlm/archive/master.zip

from languages_id import langs_id
from parameters_filtering import parameters_filtering
from badwords import badwords


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
        stopwords,
        stopwords_cutoff,
    ):
        cond = True
        if stopwords:
            words = BasicFiltering.get_words_from_sentence(sentence, strip_characters)
            stopwords_ratio = len([word for word in words if word in stopwords]) / len(
                words
            )
            cond = stopwords_ratio < stopwords_cutoff
        return cond

    @staticmethod
    def check_badwords(
        sentence,
        strip_characters,
        badwords,
        badwords_cutoff,
    ):
        cond = True
        if badwords:
            words = BasicFiltering.get_words_from_sentence(sentence, strip_characters)
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
        model_lang_id,
        lang_id_cutoff,
    ):
        cond = True
        if model_lang_id:
            words = BasicFiltering.get_words_from_sentence(sentence, strip_characters)
            sent = " ".join(words)
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
    def basic_filtering(
        sentence,
        lang_oscar_id,
        cond_remove_words_with_incorrect_substrings,
        incorrect_word_substrings,
        cond_remove_long_words,
        length_word_cutoff,
        cond_check_empty,
        strip_characters,
        cond_check_special_characters,
        special_characters,
        special_characters_cutoff,
        cond_check_stopwords,
        stopwords,
        stopwords_cutoff,
        cond_check_badwords,
        badwords,
        badwords_cutoff,
        cond_check_lang_id,
        model_lang_id,
        lang_id_cutoff,
    ):
        if cond_remove_words_with_incorrect_substrings:
            sentence = BasicFiltering.remove_words_with_incorrect_substrings(
                sentence,
                incorrect_word_substrings,
            )
        if cond_remove_long_words:
            sentence = BasicFiltering.remove_long_words(sentence, length_word_cutoff)

        if cond_check_empty:
            if not BasicFiltering.check_empty(sentence, strip_characters):
                return False
        if cond_check_special_characters:
            if not BasicFiltering.check_special_characters(
                sentence,
                special_characters,
                special_characters_cutoff,
            ):
                return False
        if cond_check_stopwords:
            if not BasicFiltering.check_stopwords(
                sentence,
                strip_characters,
                stopwords,
                stopwords_cutoff,
            ):
                return False
        if cond_check_badwords:
            if not BasicFiltering.check_badwords(
                sentence,
                strip_characters,
                badwords,
                badwords_cutoff,
            ):
                return False
        if cond_check_lang_id:
            if not BasicFiltering.check_lang_id(
                sentence,
                strip_characters,
                lang_oscar_id,
                model_lang_id,
                lang_id_cutoff,
            ):
                return False
        return True


class PerplexityFiltering:
    @staticmethod
    def get_perplexity(pp_model, doc):
        # To open a model: pp_model = kenlm.Model(path_model)
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = pp_model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return 10.0 ** (-doc_log_score / doc_length)


class OscarBasicFiltering:
    def __init__(
        self,
        lang_oscar_id,
        path_model_fasttext,
    ):
        self.lang_oscar_id = lang_oscar_id

        nltk_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "nltk_id"
        ].iloc[0]
        if nltk_lang_id:
            self.stopwords = set(nltk_stopwords.words(nltk_lang_id))
        else:
            self.stopwords = None

        badwords_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "badwords_id"
        ].iloc[0]
        if badwords_lang_id:
            self.badwords = set(badwords[badwords_lang_id])
        else:
            self.badwords = None

        fasttext_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "fasttext_id"
        ].iloc[0]
        if fasttext_lang_id:
            self.model_lang_id = fasttext.load_model(path_model_fasttext)
        else:
            self.model_lang_id = None

        if lang_oscar_id in parameters_filtering:
            self.param = parameters_filtering[lang_oscar_id]
        else:
            self.param = parameters_filtering["default"]

    def filtering(self, path_oscar_file):
        with open(
            path_oscar_file.replace(".txt.gz", "") + ".sample_filtered.txt",
            "w",
            encoding="utf8",
        ) as f:
            with gzip.open(path_oscar_file, "rb") as f2:
                from tqdm import tqdm as tqdm

                for id_, line in enumerate(tqdm(f2)):
                    line = line.decode().strip()
                    if BasicFiltering.basic_filtering(
                        sentence=line,
                        lang_oscar_id=self.lang_oscar_id,
                        cond_remove_words_with_incorrect_substrings=self.param[
                            "cond_remove_words_with_incorrect_substrings"
                        ],
                        incorrect_word_substrings=self.param[
                            "incorrect_word_substrings"
                        ],
                        cond_remove_long_words=self.param["cond_remove_long_words"],
                        length_word_cutoff=self.param["length_word_cutoff"],
                        cond_check_empty=self.param["cond_check_empty"],
                        strip_characters=self.param["strip_characters"],
                        cond_check_special_characters=self.param[
                            "cond_check_special_characters"
                        ],
                        special_characters=self.param["special_characters"],
                        special_characters_cutoff=self.param[
                            "special_characters_cutoff"
                        ],
                        cond_check_stopwords=self.param["cond_check_stopwords"],
                        stopwords=self.stopwords,
                        stopwords_cutoff=self.param["stopwords_cutoff"],
                        cond_check_badwords=self.param["cond_check_badwords"],
                        badwords=self.badwords,
                        badwords_cutoff=self.param["badwords_cutoff"],
                        cond_check_lang_id=self.param["cond_check_lang_id"],
                        model_lang_id=self.model_lang_id,
                        lang_id_cutoff=self.param["lang_id_cutoff"],
                    ):
                        f.write(line + "\n\n")
        os.unlink(path_oscar_file)


if __name__ == "__main__":
    lang_oscar_id = "en"
    path_model_fasttext = "/tmp/lid.176.bin"
    path_oscar_file = "../en_part_1.txt.gz"
    oscar_basic_filtering = OscarBasicFiltering(
        lang_oscar_id=lang_oscar_id, path_model_fasttext=path_model_fasttext
    )
    oscar_basic_filtering.filtering(path_oscar_file=path_oscar_file)

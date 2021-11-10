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
        stopwords_cutoff,
        cond_check_badwords,
        badwords_cutoff,
        cond_check_lang_id,
        path_model_fasttext,
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
                lang_oscar_id,
                stopwords_cutoff,
            ):
                return False
        if cond_check_badwords:
            if not BasicFiltering.check_badwords(
                sentence,
                strip_characters,
                lang_oscar_id,
                badwords_cutoff,
            ):
                return False
        if cond_check_lang_id:
            if not BasicFiltering.check_lang_id(
                sentence,
                strip_characters,
                lang_oscar_id,
                path_model_fasttext,
                lang_id_cutoff,
            ):
                return False
        return True


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
        path_model_fasttext,
    ):
        self.lang_oscar_id = lang_oscar_id
        self.path_model_fasttext = path_model_fasttext
        if lang_oscar_id in parameters_filtering:
            self.param = parameters_filtering[lang_oscar_id]
        else:
            self.param = parameters_filtering["en"]

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

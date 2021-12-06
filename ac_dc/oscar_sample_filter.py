import re

import fasttext

# To download the fasttext model:
# wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

import sentencepiece
import kenlm

import pathlib

from languages_id import langs_id
from parameters_filtering import parameters_filtering
from normalization import normalization
from stopwords import stopwords
from badwords import badwords


class LoadParameters:
    @staticmethod
    def load_stopwords(lang_oscar_id):
        stopwords_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "stopwords_id"
        ].iloc[0]
        if stopwords_lang_id:
            stopwords_lang = set(stopwords[stopwords_lang_id])
        else:
            stopwords_lang = None
        return stopwords_lang

    @staticmethod
    def load_badwords(lang_oscar_id):
        badwords_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "badwords_id"
        ].iloc[0]
        if badwords_lang_id:
            badwords_lang = set(badwords[badwords_lang_id])
        else:
            badwords_lang = None
        return badwords_lang

    @staticmethod
    def load_model_lang_id(lang_oscar_id, path_fasttext_model):
        fasttext_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "fasttext_id"
        ].iloc[0]
        if fasttext_lang_id:
            model_lang_id = fasttext.load_model(path_fasttext_model)
        else:
            model_lang_id = None
        return model_lang_id

    @staticmethod
    def load_sentencepiece_model(lang_oscar_id, path_sentencepiece_model):
        sentencepiece_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "sentencepiece_id"
        ].iloc[0]
        if sentencepiece_lang_id:
            sentencepiece_model = sentencepiece.SentencePieceProcessor()
            sentencepiece_model.load(path_sentencepiece_model)
        else:
            sentencepiece_model = None
        return sentencepiece_model

    @staticmethod
    def load_kenlm_model(lang_oscar_id, path_kenlm_model):
        kenlm_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "kenlm_id"
        ].iloc[0]
        if kenlm_lang_id:
            kenlm_model = kenlm.Model(path_kenlm_model)
        else:
            kenlm_model = None
        return kenlm_model

    @staticmethod
    def load_parameters(lang_oscar_id):
        if lang_oscar_id in parameters_filtering:
            param = parameters_filtering[lang_oscar_id]
        else:
            param = parameters_filtering["default"]
        return param


class ModifyingSentences:
    @staticmethod
    def remove_non_printing_characters(sentence, non_printing_characters_re):
        return non_printing_characters_re.sub("", sentence)

    @staticmethod
    def replace_digits_with_zeros(sentence, digits_re):
        return digits_re.sub("0", sentence)

    @staticmethod
    def replace_unicode_punctuation(sentence, unicode_punctuation):
        return "".join(unicode_punctuation.get(c, c) for c in sentence)

    @staticmethod
    def normalization(
        sentence,
        remove_non_printing_characters,
        strip,
        lower_case,
        replace_digits_with_zeros,
        replace_unicode_punctuation,
        non_printing_characters_re=normalization["non_printing_characters_re"],
        digits_re=normalization["digits_re"],
        unicode_punctuation=normalization["unicode_punctuation"],
    ):
        if remove_non_printing_characters:
            sentence = ModifyingSentences.remove_non_printing_characters(
                sentence, non_printing_characters_re
            )
        if strip:
            sentence = sentence.strip()
        if not sentence:
            return sentence
        if lower_case:
            sentence = sentence.lower()
        if replace_digits_with_zeros:
            sentence = ModifyingSentences.replace_digits_with_zeros(sentence, digits_re)
        if replace_unicode_punctuation:
            sentence = ModifyingSentences.replace_unicode_punctuation(
                sentence, unicode_punctuation
            )
        return sentence

    @staticmethod
    def tokenization(sentence, sentencepiece_model):
        sentence_tokenized = sentencepiece_model.encode_as_pieces(sentence)
        sentence_tokenized = " ".join(sentence_tokenized)
        return sentence_tokenized

    @staticmethod
    def get_words_from_sentence(sentence, strip_characters):
        """Get words from a sentence. Non reversible since the sentence
        is split on multiple characters and words are stripped of
        special characters. Useful to compute ratios, like the
        stopwords ratio."""
        sentence = sentence.lower()
        words = [word.strip(strip_characters) for word in re.split(" |\n|\t", sentence)]
        words = [word for word in words if word]
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
        length_word_max_cutoff,
    ):
        words = sentence.split(" ")
        words = [word for word in words if len(word) <= length_word_max_cutoff]
        filtered_sentence = " ".join(words)
        return filtered_sentence

    @staticmethod
    def modifying_sentences(
        sentence,
        cond_replace_unicode_punctuation,
        cond_remove_words_with_incorrect_substrings,
        incorrect_word_substrings,
        cond_remove_long_words,
        length_word_max_cutoff,
    ):
        sentence = ModifyingSentences.normalization(
            sentence=sentence,
            remove_non_printing_characters=False,
            strip=True,
            lower_case=False,
            replace_digits_with_zeros=False,
            replace_unicode_punctuation=cond_replace_unicode_punctuation,
        )
        if cond_remove_words_with_incorrect_substrings:
            sentence = ModifyingSentences.remove_words_with_incorrect_substrings(
                sentence,
                incorrect_word_substrings,
            )
        if cond_remove_long_words:
            sentence = ModifyingSentences.remove_long_words(
                sentence, length_word_max_cutoff
            )
        return sentence


class OscarModifyingSentences:
    def __init__(self, lang_oscar_id):
        self.lang_oscar_id = lang_oscar_id
        self.param = LoadParameters.load_parameters(lang_oscar_id)

    def __call__(self, example):
        example["text"] = ModifyingSentences.modifying_sentences(
            sentence=example["text"],
            cond_replace_unicode_punctuation=self.param["cond_replace_unicode_punctuation"],
            cond_remove_words_with_incorrect_substrings=self.param[
                "cond_remove_words_with_incorrect_substrings"
            ],
            incorrect_word_substrings=self.param["incorrect_word_substrings"],
            cond_remove_long_words=self.param["cond_remove_long_words"],
            length_word_max_cutoff=self.param["length_word_max_cutoff"],
        )
        return example

    def __reduce__(self):
        return (self.__class__, (self.lang_oscar_id,))


class Filtering:
    @staticmethod
    def check_empty(sentence, strip_characters):
        words = ModifyingSentences.get_words_from_sentence(sentence, strip_characters)
        cond = len(words) > 0
        return cond

    @staticmethod
    def compute_special_characters_ratio(sentence, special_characters):
        set_special_characters = {char for char in special_characters}
        special_characters_ratio = len(
            [char for char in sentence if char in set_special_characters]
        ) / len(sentence)
        return special_characters_ratio

    @staticmethod
    def check_special_characters(
        sentence,
        special_characters,
        special_characters_max_cutoff,
    ):
        special_characters_ratio = Filtering.compute_special_characters_ratio(
            sentence, special_characters
        )
        cond = special_characters_ratio <= special_characters_max_cutoff
        return cond

    @staticmethod
    def compute_stopwords_ratio(sentence, strip_characters, stopwords):
        words = ModifyingSentences.get_words_from_sentence(sentence, strip_characters)
        stopwords_ratio = len([word for word in words if word in stopwords]) / len(
            words
        )
        return stopwords_ratio

    @staticmethod
    def check_stopwords(
        sentence,
        strip_characters,
        stopwords,
        stopwords_min_cutoff,
    ):
        cond = True
        if stopwords:
            stopwords_ratio = Filtering.compute_stopwords_ratio(
                sentence, strip_characters, stopwords
            )
            cond = stopwords_ratio >= stopwords_min_cutoff
        return cond

    @staticmethod
    def compute_badwords_ratio(sentence, strip_characters, badwords):
        words = ModifyingSentences.get_words_from_sentence(sentence, strip_characters)
        badwords_ratio = len([word for word in words if word in badwords]) / len(words)
        return badwords_ratio

    @staticmethod
    def check_badwords(
        sentence,
        strip_characters,
        badwords,
        badwords_max_cutoff,
    ):
        cond = True
        if badwords:
            badwords_ratio = Filtering.compute_badwords_ratio(
                sentence, strip_characters, badwords
            )
            cond = badwords_ratio <= badwords_max_cutoff
        return cond

    @staticmethod
    def compute_lang_id_pred_score(sentence, model_lang_id):
        sentence = sentence.lower().replace("\n", " ")
        pred = model_lang_id.predict(sentence)
        lang_pred_fasttext_id = pred[0][0].replace("__label__", "")
        score_pred = pred[1][0]
        lang_pred_oscar_id = langs_id.loc[
            langs_id["fasttext_id"] == lang_pred_fasttext_id, "oscar_id"
        ]
        if len(lang_pred_oscar_id) > 0:
            lang_pred_oscar_id = lang_pred_oscar_id.iloc[0]
        else:
            lang_pred_oscar_id = "unknown"
        return lang_pred_oscar_id, score_pred

    @staticmethod
    def check_lang_id(
        sentence,
        lang_oscar_id,
        model_lang_id,
        lang_id_min_cutoff,
    ):
        cond = True
        if model_lang_id:
            lang_pred_oscar_id, score_pred = Filtering.compute_lang_id_pred_score(
                sentence, model_lang_id
            )
            cond = (lang_pred_oscar_id == lang_oscar_id) and (
                score_pred >= lang_id_min_cutoff
            )
        return cond

    @staticmethod
    def compute_perplexity_score(doc, sentencepiece_model, kenlm_model):
        doc = ModifyingSentences.normalization(
            sentence=doc,
            remove_non_printing_characters=True,
            strip=True,
            lower_case=True,
            replace_digits_with_zeros=True,
            replace_unicode_punctuation=True,
        )
        doc = ModifyingSentences.tokenization(doc, sentencepiece_model)
        doc_log_score, doc_length = 0, 0
        for line in doc.split("\n"):
            log_score = kenlm_model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        pp_score = 10.0 ** (-doc_log_score / doc_length)
        pp_score = round(pp_score, 1)
        return pp_score

    @staticmethod
    def check_perplexity(
        sentence,
        sentencepiece_model,
        kenlm_model,
        perplexity_max_cutoff,
    ):
        cond = True
        if kenlm_model:
            score = Filtering.compute_perplexity_score(
                sentence, sentencepiece_model, kenlm_model
            )
            cond = score <= perplexity_max_cutoff
        return cond

    @staticmethod
    def filtering(
        sentence,
        cond_check_empty,
        strip_characters,
        cond_check_special_characters,
        special_characters,
        special_characters_max_cutoff,
        cond_check_stopwords,
        stopwords,
        stopwords_min_cutoff,
        cond_check_badwords,
        badwords,
        badwords_max_cutoff,
        cond_check_lang_id,
        lang_oscar_id,
        model_lang_id,
        lang_id_min_cutoff,
        cond_check_perplexity,
        sentencepiece_model,
        kenlm_model,
        perplexity_max_cutoff,
    ):
        if cond_check_empty:
            if not Filtering.check_empty(sentence, strip_characters):
                return False
        if cond_check_special_characters:
            if not Filtering.check_special_characters(
                sentence,
                special_characters,
                special_characters_max_cutoff,
            ):
                return False
        if cond_check_stopwords:
            if not Filtering.check_stopwords(
                sentence,
                strip_characters,
                stopwords,
                stopwords_min_cutoff,
            ):
                return False
        if cond_check_badwords:
            if not Filtering.check_badwords(
                sentence,
                strip_characters,
                badwords,
                badwords_max_cutoff,
            ):
                return False
        if cond_check_lang_id:
            if not Filtering.check_lang_id(
                sentence,
                lang_oscar_id,
                model_lang_id,
                lang_id_min_cutoff,
            ):
                return False
        if cond_check_perplexity:
            if not Filtering.check_perplexity(
                sentence,
                sentencepiece_model,
                kenlm_model,
                perplexity_max_cutoff,
            ):
                return False
        return True


class FuncOscarFiltering:
    def __init__(
        self,
        lang_oscar_id,
        path_fasttext_model,
        path_sentencepiece_model,
        path_kenlm_model,
    ):
        self.lang_oscar_id = lang_oscar_id
        self.path_fasttext_model = path_fasttext_model
        self.path_sentencepiece_model = path_sentencepiece_model
        self.path_kenlm_model = path_kenlm_model

        self.stopwords = LoadParameters.load_stopwords(lang_oscar_id)
        self.badwords = LoadParameters.load_badwords(lang_oscar_id)
        self.model_lang_id = LoadParameters.load_model_lang_id(
            lang_oscar_id, path_fasttext_model
        )
        self.sentencepiece_model = LoadParameters.load_sentencepiece_model(
            lang_oscar_id, path_sentencepiece_model
        )
        self.kenlm_model = LoadParameters.load_kenlm_model(
            lang_oscar_id, path_kenlm_model
        )
        self.param = LoadParameters.load_parameters(lang_oscar_id)

    def __call__(self, example):
        keep_example = Filtering.filtering(
            sentence=example["text"].strip(),
            cond_check_empty=self.param["cond_check_empty"],
            strip_characters=self.param["strip_characters"],
            cond_check_special_characters=self.param["cond_check_special_characters"],
            special_characters=self.param["special_characters"],
            special_characters_max_cutoff=self.param["special_characters_max_cutoff"],
            cond_check_stopwords=self.param["cond_check_stopwords"],
            stopwords=self.stopwords,
            stopwords_min_cutoff=self.param["stopwords_min_cutoff"],
            cond_check_badwords=self.param["cond_check_badwords"],
            badwords=self.badwords,
            badwords_max_cutoff=self.param["badwords_max_cutoff"],
            cond_check_lang_id=self.param["cond_check_lang_id"],
            lang_oscar_id=self.lang_oscar_id,
            model_lang_id=self.model_lang_id,
            lang_id_min_cutoff=self.param["lang_id_min_cutoff"],
            cond_check_perplexity=self.param["cond_check_perplexity"],
            sentencepiece_model=self.sentencepiece_model,
            kenlm_model=self.kenlm_model,
            perplexity_max_cutoff=self.param["perplexity_max_cutoff"],
        )
        return keep_example

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.lang_oscar_id,
                self.path_fasttext_model,
                self.path_sentencepiece_model,
                self.path_kenlm_model,
            ),
        )


class OscarFiltering:
    def __init__(
        self,
        dataset,
        lang_oscar_id,
        path_fasttext_model,
        path_sentencepiece_model,
        path_kenlm_model,
        num_proc,
        path_dir_save_oscar,
    ):
        self.ds = dataset
        self.lang_oscar_id = lang_oscar_id
        self.path_fasttext_model = path_fasttext_model
        self.path_sentencepiece_model = path_sentencepiece_model
        self.path_kenlm_model = path_kenlm_model
        self.num_proc = num_proc
        self.path_dir_save_oscar = path_dir_save_oscar

    def modifying_sentences(self):
        oscar_modifying_sentences = OscarModifyingSentences(self.lang_oscar_id)
        self.ds = self.ds.map(oscar_modifying_sentences, num_proc=self.num_proc)

    def filtering(self):
        func_oscar_filtering = FuncOscarFiltering(
            self.lang_oscar_id,
            self.path_fasttext_model,
            self.path_sentencepiece_model,
            self.path_kenlm_model,
        )
        self.ds = self.ds.filter(func_oscar_filtering, num_proc=self.num_proc)

    def save_dataset(self):
        pathlib.Path(self.path_dir_save_oscar).mkdir(parents=True, exist_ok=True)
        path_dir_save_dataset = pathlib.PurePath(
            self.path_dir_save_oscar, self.lang_oscar_id
        )
        pathlib.Path(path_dir_save_dataset).mkdir(parents=True, exist_ok=True)
        self.ds.save_to_disk(path_dir_save_dataset)

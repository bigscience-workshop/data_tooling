from datasets import load_dataset

import fasttext

# To download the fasttext model:
# wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

import kenlm  # pip install https://github.com/kpu/kenlm/archive/master.zip

import pathlib

from languages_id import langs_id
from parameters_filtering import parameters_filtering
from stopwords import stopwords
from badwords import badwords


class LoadParameters:
    @staticmethod
    def load_stopwords(lang_oscar_id):
        nltk_lang_id = langs_id.loc[
            langs_id["oscar_id"] == lang_oscar_id, "nltk_id"
        ].iloc[0]
        if nltk_lang_id:
            stopwords_lang = set(stopwords[nltk_lang_id])
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
    def lower_strip_sentence(sentence):
        sent = sentence.lower().strip()
        return sent

    @staticmethod
    def get_words_from_sentence(sentence, strip_characters):
        sent = ModifyingSentences.lower_strip_sentence(sentence)
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
    def modifying_sentences(
        sentence,
        cond_remove_words_with_incorrect_substrings,
        incorrect_word_substrings,
        cond_remove_long_words,
        length_word_cutoff,
    ):
        if cond_remove_words_with_incorrect_substrings:
            sentence = ModifyingSentences.remove_words_with_incorrect_substrings(
                sentence,
                incorrect_word_substrings,
            )
        if cond_remove_long_words:
            sentence = ModifyingSentences.remove_long_words(
                sentence, length_word_cutoff
            )
        return sentence


class OscarModifyingSentences:
    def __init__(self, lang_oscar_id):
        self.lang_oscar_id = lang_oscar_id
        self.param = LoadParameters.load_parameters(lang_oscar_id)

    def __call__(self, example):
        example["text"] = ModifyingSentences.modifying_sentences(
            sentence=example["text"],
            cond_remove_words_with_incorrect_substrings=self.param[
                "cond_remove_words_with_incorrect_substrings"
            ],
            incorrect_word_substrings=self.param["incorrect_word_substrings"],
            cond_remove_long_words=self.param["cond_remove_long_words"],
            length_word_cutoff=self.param["length_word_cutoff"],
        )
        return example

    def __reduce__(self):
        return (self.__class__, (self.lang_oscar_id,))


class Filtering:
    @staticmethod
    def check_empty(sentence, strip_characters):
        sent = ModifyingSentences.lower_strip_sentence(sentence)
        words = ModifyingSentences.get_words_from_sentence(sentence, strip_characters)
        cond = (len(sent) > 0) and (len(words) > 0)
        return cond

    @staticmethod
    def compute_special_characters_ratio(sentence, special_characters):
        sent = ModifyingSentences.lower_strip_sentence(sentence)
        set_special_characters = {char for char in special_characters}
        special_characters_ratio = len(
            [char for char in sent if char in set_special_characters]
        ) / len(sent)
        return special_characters_ratio

    @staticmethod
    def check_special_characters(
        sentence,
        special_characters,
        special_characters_cutoff,
    ):
        special_characters_ratio = Filtering.compute_special_characters_ratio(
            sentence, special_characters
        )
        cond = special_characters_ratio < special_characters_cutoff
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
        stopwords_max_cutoff,
    ):
        cond = True
        if stopwords:
            stopwords_ratio = Filtering.compute_stopwords_ratio(
                sentence, strip_characters, stopwords
            )
            cond = (stopwords_ratio > stopwords_min_cutoff) and (
                stopwords_ratio < stopwords_max_cutoff
            )
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
        badwords_cutoff,
    ):
        cond = True
        if badwords:
            badwords_ratio = Filtering.compute_badwords_ratio(
                sentence, strip_characters, badwords
            )
            cond = badwords_ratio < badwords_cutoff
        return cond

    @staticmethod
    def compute_lang_id_pred_score(sentence, strip_characters, model_lang_id):
        words = ModifyingSentences.get_words_from_sentence(sentence, strip_characters)
        sent = " ".join(words).replace("\n", " ")
        pred = model_lang_id.predict(sent)
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
        strip_characters,
        lang_oscar_id,
        model_lang_id,
        lang_id_cutoff,
    ):
        cond = True
        if model_lang_id:
            lang_pred_oscar_id, score_pred = Filtering.compute_lang_id_pred_score(
                sentence, strip_characters, model_lang_id
            )
            cond = (lang_pred_oscar_id == lang_oscar_id) and (
                score_pred > lang_id_cutoff
            )
        return cond

    @staticmethod
    def compute_perplexity_score(sentence, kenlm_model):
        doc_log_score, doc_length = 0, 0
        for line in sentence.split("\n"):
            log_score = kenlm_model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length
        return 10.0 ** (-doc_log_score / doc_length)

    @staticmethod
    def check_perplexity(
        sentence,
        kenlm_model,
        perplexity_cutoff,
    ):
        cond = True
        if kenlm_model:
            score = Filtering.compute_perplexity_score(sentence, kenlm_model)
            cond = score < perplexity_cutoff
        return cond

    @staticmethod
    def filtering(
        sentence,
        cond_check_empty,
        strip_characters,
        cond_check_special_characters,
        special_characters,
        special_characters_cutoff,
        cond_check_stopwords,
        stopwords,
        stopwords_min_cutoff,
        stopwords_max_cutoff,
        cond_check_badwords,
        badwords,
        badwords_cutoff,
        cond_check_lang_id,
        lang_oscar_id,
        model_lang_id,
        lang_id_cutoff,
        cond_check_perplexity,
        kenlm_model,
        perplexity_cutoff,
    ):
        if cond_check_empty:
            if not Filtering.check_empty(sentence, strip_characters):
                return False
        if cond_check_special_characters:
            if not Filtering.check_special_characters(
                sentence,
                special_characters,
                special_characters_cutoff,
            ):
                return False
        if cond_check_stopwords:
            if not Filtering.check_stopwords(
                sentence,
                strip_characters,
                stopwords,
                stopwords_min_cutoff,
                stopwords_max_cutoff,
            ):
                return False
        if cond_check_badwords:
            if not Filtering.check_badwords(
                sentence,
                strip_characters,
                badwords,
                badwords_cutoff,
            ):
                return False
        if cond_check_lang_id:
            if not Filtering.check_lang_id(
                sentence,
                strip_characters,
                lang_oscar_id,
                model_lang_id,
                lang_id_cutoff,
            ):
                return False
        if cond_check_perplexity:
            if not Filtering.check_perplexity(
                sentence,
                kenlm_model,
                perplexity_cutoff,
            ):
                return False
        return True


class FuncOscarFiltering:
    def __init__(self, lang_oscar_id, path_fasttext_model, path_kenlm_model):
        self.lang_oscar_id = lang_oscar_id
        self.path_fasttext_model = path_fasttext_model
        self.path_kenlm_model = path_kenlm_model

        self.stopwords = LoadParameters.load_stopwords(lang_oscar_id)
        self.badwords = LoadParameters.load_badwords(lang_oscar_id)
        self.model_lang_id = LoadParameters.load_model_lang_id(
            lang_oscar_id, path_fasttext_model
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
            special_characters_cutoff=self.param["special_characters_cutoff"],
            cond_check_stopwords=self.param["cond_check_stopwords"],
            stopwords=self.stopwords,
            stopwords_min_cutoff=self.param["stopwords_min_cutoff"],
            stopwords_max_cutoff=self.param["stopwords_max_cutoff"],
            cond_check_badwords=self.param["cond_check_badwords"],
            badwords=self.badwords,
            badwords_cutoff=self.param["badwords_cutoff"],
            cond_check_lang_id=self.param["cond_check_lang_id"],
            lang_oscar_id=self.lang_oscar_id,
            model_lang_id=self.model_lang_id,
            lang_id_cutoff=self.param["lang_id_cutoff"],
            cond_check_perplexity=self.param["cond_check_perplexity"],
            kenlm_model=self.kenlm_model,
            perplexity_cutoff=self.param["perplexity_cutoff"],
        )
        return keep_example

    def __reduce__(self):
        return (
            self.__class__,
            (self.lang_oscar_id, self.path_fasttext_model, self.path_kenlm_model),
        )


class OscarFiltering:
    def __init__(
        self,
        lang_oscar_id,
        path_fasttext_model,
        path_kenlm_model,
        num_proc,
        path_dir_save_oscar,
        use_registry,
        registry_data_files,
    ):
        self.lang_oscar_id = lang_oscar_id
        self.path_fasttext_model = path_fasttext_model
        self.path_kenlm_model = path_kenlm_model
        if use_registry:
            data_files = registry_data_files if registry_data_files else f"{lang_oscar_id}/*.jsonl.gz"
            self.ds = load_dataset(
                "mhtoin/register_oscar", data_files=data_files,
            )["train"]
        else:
            self.ds = load_dataset(
                "oscar", f"unshuffled_deduplicated_{self.lang_oscar_id}"
            )["train"]
        self.num_proc = num_proc
        self.path_dir_save_oscar = path_dir_save_oscar

    def modifying_sentences(self):
        oscar_modifying_sentences = OscarModifyingSentences(self.lang_oscar_id)
        self.ds = self.ds.map(oscar_modifying_sentences, num_proc=self.num_proc)

    def filtering(self):
        func_oscar_filtering = FuncOscarFiltering(
            self.lang_oscar_id, self.path_fasttext_model, self.path_kenlm_model
        )
        self.ds = self.ds.filter(func_oscar_filtering, num_proc=self.num_proc)

    def save_dataset(self):
        pathlib.Path(self.path_dir_save_oscar).mkdir(parents=True, exist_ok=True)
        path_dir_save_dataset = pathlib.PurePath(
            self.path_dir_save_oscar, self.lang_oscar_id
        )
        pathlib.Path(path_dir_save_dataset).mkdir(parents=True, exist_ok=True)
        self.ds.save_to_disk(path_dir_save_dataset)

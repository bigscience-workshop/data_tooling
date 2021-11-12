"""Basic filtering of garbage and perplexity filtering for OSCAR v1."""

import argparse
import pathlib

from datasets import load_dataset

import nltk
import numpy as np

nltk.download("stopwords")
from nltk.corpus import stopwords as nltk_stopwords

import fasttext

np.random.seed(1337)

# To download the fasttext model:
# wget -O /tmp/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# import kenlm  # pip install https://github.com/kpu/kenlm/archive/master.zip

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
    def modifying_sentences(
        sentence,
        cond_remove_words_with_incorrect_substrings,
        incorrect_word_substrings,
        cond_remove_long_words,
        length_word_cutoff,
    ):
        if cond_remove_words_with_incorrect_substrings:
            sentence = BasicFiltering.remove_words_with_incorrect_substrings(
                sentence,
                incorrect_word_substrings,
            )
        if cond_remove_long_words:
            sentence = BasicFiltering.remove_long_words(sentence, length_word_cutoff)
        return sentence

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
        ) / (len(sent) + 1e-10)
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
            sent = " ".join(words).replace("\n", " ")
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
    def basic_filtering_debug(
        example,
        lang_oscar_id,
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
        sentence = example["text"].strip()
        if cond_check_empty:
            example["cond_check_empty"] = BasicFiltering.check_empty(
                sentence, strip_characters
            )

        if cond_check_special_characters:
            example[
                "cond_check_special_characters"
            ] = BasicFiltering.check_special_characters(
                sentence,
                special_characters,
                special_characters_cutoff,
            )

        if cond_check_stopwords:
            example["cond_check_stopwords"] = BasicFiltering.check_stopwords(
                sentence,
                strip_characters,
                stopwords,
                stopwords_cutoff,
            )

        if cond_check_badwords:
            example["cond_check_badwords"] = BasicFiltering.check_badwords(
                sentence,
                strip_characters,
                badwords,
                badwords_cutoff,
            )

        if cond_check_lang_id:
            example["cond_check_lang_id"] = BasicFiltering.check_lang_id(
                sentence,
                strip_characters,
                lang_oscar_id,
                model_lang_id,
                lang_id_cutoff,
            )

        return example

    @staticmethod
    def basic_filtering(
        sentence,
        lang_oscar_id,
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
        num_proc,
        sample_subset,
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

        self.ds = load_dataset(
            "oscar", f"unshuffled_deduplicated_{self.lang_oscar_id}"
        )["train"]
        if sample_subset:
            subset_idx = np.random.randint(0, len(self.ds), sample_subset)
            self.ds = self.ds.select(subset_idx)
        self.num_proc = num_proc

    def basic_filtering(self, debug: bool = False):
        def func_modifying_sentences(example):
            example["text"] = BasicFiltering.modifying_sentences(
                sentence=example["text"],
                cond_remove_words_with_incorrect_substrings=self.param[
                    "cond_remove_words_with_incorrect_substrings"
                ],
                incorrect_word_substrings=self.param["incorrect_word_substrings"],
                cond_remove_long_words=self.param["cond_remove_long_words"],
                length_word_cutoff=self.param["length_word_cutoff"],
            )
            return example

        self.ds = self.ds.map(func_modifying_sentences, num_proc=self.num_proc)

        func_basic_filtering = lambda example: BasicFiltering.basic_filtering(
            sentence=example["text"].strip(),
            lang_oscar_id=self.lang_oscar_id,
            cond_check_empty=self.param["cond_check_empty"],
            strip_characters=self.param["strip_characters"],
            cond_check_special_characters=self.param["cond_check_special_characters"],
            special_characters=self.param["special_characters"],
            special_characters_cutoff=self.param["special_characters_cutoff"],
            cond_check_stopwords=self.param["cond_check_stopwords"],
            stopwords=self.stopwords,
            stopwords_cutoff=self.param["stopwords_cutoff"],
            cond_check_badwords=self.param["cond_check_badwords"],
            badwords=self.badwords,
            badwords_cutoff=self.param["badwords_cutoff"],
            cond_check_lang_id=self.param["cond_check_lang_id"],
            model_lang_id=self.model_lang_id,
            lang_id_cutoff=self.param["lang_id_cutoff"],
        )
        if debug:
            debug_func_basic_filtering = (
                lambda example: BasicFiltering.basic_filtering_debug(
                    example=example,
                    lang_oscar_id=self.lang_oscar_id,
                    cond_check_empty=self.param["cond_check_empty"],
                    strip_characters=self.param["strip_characters"],
                    cond_check_special_characters=self.param[
                        "cond_check_special_characters"
                    ],
                    special_characters=self.param["special_characters"],
                    special_characters_cutoff=self.param["special_characters_cutoff"],
                    cond_check_stopwords=self.param["cond_check_stopwords"],
                    stopwords=self.stopwords,
                    stopwords_cutoff=self.param["stopwords_cutoff"],
                    cond_check_badwords=self.param["cond_check_badwords"],
                    badwords=self.badwords,
                    badwords_cutoff=self.param["badwords_cutoff"],
                    cond_check_lang_id=self.param["cond_check_lang_id"],
                    model_lang_id=self.model_lang_id,
                    lang_id_cutoff=self.param["lang_id_cutoff"],
                )
            )

            # save Dataset with conditions for analysis
            self.ds_debug = self.ds.map(
                debug_func_basic_filtering, num_proc=self.num_proc
            )

        self.ds = self.ds.filter(func_basic_filtering, num_proc=self.num_proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Basic filtering of garbage for OSCAR v1."
    )
    parser.add_argument(
        "--lang_oscar_id",
        type=str,
        default="af",
        help="ID of the language Oscar is filtered on.",
    )
    parser.add_argument(
        "--path_model_fasttext",
        type=str,
        default="/tmp/lid.176.bin",
        help="Path to the Fasttext model used for language identification.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes for multiprocessing.",
    )
    parser.add_argument(
        "--path_dir_save_oscar",
        type=str,
        default="../Oscar_filtered/",
        help="Path to the directory where the filtered version of Oscar will be saved.",
    )
    parser.add_argument(
        "--save_debug_dataset",
        type=bool,
        default=False,
        help="Whether to save the dataset with filtering conditions for debugging.",
    )
    parser.add_argument(
        "--sample_subset",
        type=int,
        default=None,
        help="Number of lines to randomly sample.",
    )
    args = parser.parse_args()

    pathlib.Path(args.path_dir_save_oscar).mkdir(parents=True, exist_ok=True)
    path_dir_save_dataset = pathlib.PurePath(
        args.path_dir_save_oscar, args.lang_oscar_id
    )
    pathlib.Path(path_dir_save_dataset).mkdir(parents=True, exist_ok=True)

    oscar_basic_filtering = OscarBasicFiltering(
        lang_oscar_id=args.lang_oscar_id,
        path_model_fasttext=args.path_model_fasttext,
        num_proc=args.num_proc,
        sample_subset=args.sample_subset,
    )
    oscar_basic_filtering.basic_filtering(debug=args.save_debug_dataset)
    oscar_basic_filtering.ds.save_to_disk(path_dir_save_dataset)
    if args.save_debug_dataset:
        path_dir_save_dataset_discarded = pathlib.PurePath(
            args.path_dir_save_oscar, args.lang_oscar_id + "_debug"
        )
        pathlib.Path(path_dir_save_dataset_discarded).mkdir(parents=True, exist_ok=True)
        oscar_basic_filtering.ds_debug.save_to_disk(path_dir_save_dataset_discarded)

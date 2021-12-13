from datasets import load_dataset
from tqdm import tqdm
import json

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from oscar_sample_filter import LoadParameters, ModifyingSentences, Filtering


class GetDataForVisualization:
    def __init__(
        self,
        dataset,
        num_iter,
        lang_oscar_id,
        path_fasttext_model,
        path_sentencepiece_model,
        path_kenlm_model,
        path_save_stats,
    ):

        self.ds = dataset
        self.num_iter = num_iter

        self.lang_oscar_id = lang_oscar_id

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

        self.keys_stats = [
            "special_characters_ratio",
            "stopwords_ratio",
            "badwords_ratio",
            "lang_id_score",
            "perplexity_score",
        ]
        self.path_save_stats = path_save_stats

    def compute_stats(self):
        dataset = iter(self.ds)
        stats = []
        num_iter_examples = True
        for i in tqdm(range(self.num_iter)):
            stats_sentence = {}

            try:
                sentence = next(dataset)["text"]

                words = ModifyingSentences.get_words_from_sentence(
                    sentence, self.param["strip_characters"]
                )
                words = [
                    {
                        "len_word": len(word),
                        "incorrect_substring": any(
                            [
                                (i_substr in word)
                                for i_substr in self.param["incorrect_word_substrings"]
                            ]
                        ),
                        "word": word,
                    }
                    for word in words
                ]

                stats_sentence["words"] = words

                number_words = len(words)
                stats_sentence["number_words"] = number_words

                special_characters_ratio = Filtering.compute_special_characters_ratio(
                    sentence, self.param["special_characters"]
                )
                stats_sentence["special_characters_ratio"] = special_characters_ratio

                if self.stopwords:
                    stopwords_ratio = Filtering.compute_stopwords_ratio(
                        sentence, self.param["strip_characters"], self.stopwords
                    )
                    stats_sentence["stopwords_ratio"] = stopwords_ratio

                if self.badwords:
                    badwords_ratio = Filtering.compute_badwords_ratio(
                        sentence, self.param["strip_characters"], self.badwords
                    )
                    stats_sentence["badwords_ratio"] = badwords_ratio

                if self.model_lang_id:
                    _, lang_id_score = Filtering.compute_lang_id_pred_score(
                        sentence, self.model_lang_id
                    )
                    stats_sentence["lang_id_score"] = lang_id_score

                if self.kenlm_model:
                    perplexity_score = Filtering.compute_perplexity_score(
                        sentence, self.sentencepiece_model, self.kenlm_model
                    )
                    stats_sentence["perplexity_score"] = perplexity_score

                stats_sentence["text"] = sentence

                stats.append(stats_sentence)

            except:
                num_iter_examples = False

        if not num_iter_examples:
            print("Warning: num_iter is greater than the size of the dataset.")

        self.stats = stats

        with open(self.path_save_stats, "w") as f:
            json.dump(self.stats, f)


if __name__ == "__main__":

    dataset_name = "oscar"
    config_name = "unshuffled_deduplicated_en"
    data_files = None
    split = "train"
    num_iter = 15000

    dataset = load_dataset(
        dataset_name,
        config_name,
        data_files=data_files,
        split=split,
        streaming=True,
    ).shuffle(buffer_size=num_iter, seed=42)

    lang_oscar_id = "en"
    path_fasttext_model = "ac_dc/lid.176.bin"
    path_sentencepiece_model = f"ac_dc/en.sp.model"
    path_kenlm_model = f"ac_dc/en.arpa.bin"
    path_save_stats = f"ac_dc/visualization/en_examples_with_stats.json"

    get_data_for_visualization = GetDataForVisualization(
        dataset,
        num_iter,
        lang_oscar_id,
        path_fasttext_model,
        path_sentencepiece_model,
        path_kenlm_model,
        path_save_stats,
    )
    get_data_for_visualization.compute_stats()

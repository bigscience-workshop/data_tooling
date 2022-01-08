from datasets import load_dataset
from tqdm import tqdm
import json

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from filtering import LoadParameters, ModifyingDocuments, Filtering


class GetDataForVisualization:
    def __init__(
        self,
        dataset,
        num_iter,
        lang_dataset_id,
        path_fasttext_model,
        path_sentencepiece_model,
        path_kenlm_model,
        path_save_stats,
    ):

        self.ds = dataset
        self.num_iter = num_iter

        self.lang_dataset_id = lang_dataset_id

        self.param = LoadParameters.load_parameters(lang_dataset_id)
        self.stopwords = LoadParameters.load_stopwords(lang_dataset_id)
        self.badwords = LoadParameters.load_badwords(lang_dataset_id)
        self.model_lang_id = LoadParameters.load_model_lang_id(
            lang_dataset_id, path_fasttext_model
        )
        self.sentencepiece_model = LoadParameters.load_sentencepiece_model(
            lang_dataset_id, path_sentencepiece_model
        )
        self.sentencepiece_model_tok = (
            self.sentencepiece_model if self.param["tokenization"] else None
        )
        self.kenlm_model = LoadParameters.load_kenlm_model(
            lang_dataset_id, path_kenlm_model
        )

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
            stats_document = {}

            try:
                document = next(dataset)["text"]

                words = ModifyingDocuments.get_words_from_document(
                    document,
                    sentencepiece_model_tok=self.sentencepiece_model_tok,
                    lower_case=True,
                    strip_characters=self.param["strip_characters"],
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

                if not self.param["tokenization"]:
                    stats_document["words"] = words

                number_words = len(words)
                stats_document["number_words"] = number_words

                repetitions_ratios = {
                    n: round(Filtering.compute_repetitions_ratio(document, n), 4)
                    for n in range(2, 16)
                }
                stats_document["repetitions_ratio"] = repetitions_ratios

                special_characters_ratio = Filtering.compute_special_characters_ratio(
                    document, self.param["special_characters"]
                )
                stats_document["special_characters_ratio"] = special_characters_ratio

                if self.stopwords:
                    stopwords_ratio = Filtering.compute_stopwords_ratio(
                        document,
                        self.sentencepiece_model_tok,
                        self.param["strip_characters"],
                        self.param["cond_words_augmentation"],
                        self.param["words_augmentation_group_sizes"],
                        self.param["words_augmentation_join_char"],
                        self.stopwords,
                    )
                    stats_document["stopwords_ratio"] = stopwords_ratio

                if self.badwords:
                    badwords_ratio = Filtering.compute_badwords_ratio(
                        document,
                        self.sentencepiece_model_tok,
                        self.param["strip_characters"],
                        self.param["cond_words_augmentation"],
                        self.param["words_augmentation_group_sizes"],
                        self.param["words_augmentation_join_char"],
                        self.badwords,
                    )
                    stats_document["badwords_ratio"] = badwords_ratio

                if self.model_lang_id:
                    _, lang_id_score = Filtering.compute_lang_id_pred_score(
                        document, self.model_lang_id
                    )
                    stats_document["lang_id_score"] = lang_id_score

                if self.kenlm_model:
                    perplexity_score = Filtering.compute_perplexity_score(
                        document, self.sentencepiece_model, self.kenlm_model
                    )
                    stats_document["perplexity_score"] = perplexity_score

                stats_document["text"] = document

                stats.append(stats_document)

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

    lang_dataset_id = "en"
    path_fasttext_model = "ac_dc/lid.176.bin"
    path_sentencepiece_model = f"ac_dc/en.sp.model"
    path_kenlm_model = f"ac_dc/en.arpa.bin"
    path_save_stats = f"ac_dc/visualization/en_examples_with_stats.json"

    dataset = load_dataset(
        dataset_name,
        config_name,
        data_files=data_files,
        split=split,
        streaming=True,
    ).shuffle(buffer_size=num_iter, seed=42)

    get_data_for_visualization = GetDataForVisualization(
        dataset,
        num_iter,
        lang_dataset_id,
        path_fasttext_model,
        path_sentencepiece_model,
        path_kenlm_model,
        path_save_stats,
    )
    get_data_for_visualization.compute_stats()

from datasets import load_dataset
from tqdm import tqdm
import json

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from oscar_sample_filter import LoadParameters, Filtering


class GetDataForVisualization:
    def __init__(
        self,
        dataset,
        num_iter,
        lang_oscar_id,
        path_fasttext_model,
        path_kenlm_model,
        path_sentence_piece_model,
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
        self.kenlm_model = LoadParameters.load_kenlm_model(
            lang_oscar_id, path_kenlm_model, path_sentence_piece_model
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
                        sentence, self.param["strip_characters"], self.model_lang_id
                    )
                    stats_sentence["lang_id_score"] = lang_id_score

                if self.kenlm_model:
                    perplexity_score = Filtering.compute_perplexity_score(
                        sentence, self.kenlm_model
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
    config_name = "unshuffled_deduplicated_af"
    data_files = None
    split = "train"
    num_iter = 5000

    dataset = load_dataset(
        dataset_name,
        config_name,
        data_files=data_files,
        split=split,
        streaming=True,
    ).shuffle(buffer_size=num_iter, seed=42)

    lang_oscar_id = "af"
    path_fasttext_model = "/tmp/lid.176.bin"
    path_kenlm_model = f"ac_dc/af.arpa.bin"
    path_sentence_piece_model = f"ac_dc/af.sp.model"
    path_save_stats = f"./af_examples_with_stats.json"

    get_data_for_visualization = GetDataForVisualization(
        dataset,
        num_iter,
        lang_oscar_id,
        path_fasttext_model,
        path_kenlm_model,
        path_sentence_piece_model,
        path_save_stats,
    )
    get_data_for_visualization.compute_stats()

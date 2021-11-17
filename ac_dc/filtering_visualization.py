from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm

from oscar_sample_filter import LoadParameters, Filtering


class FilteringVisualization:
    def __init__(
        self,
        lang_oscar_id,
        path_fasttext_model,
        path_kenlm_model,
        path_sentence_piece_model,
        num_iter,
        path_dir_save_visu,
    ):
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

        self.ds = load_dataset(
            "oscar",
            f"unshuffled_deduplicated_{lang_oscar_id}",
            split="train",
        )
        self.num_iter = min(num_iter, self.ds.num_rows)
        self.path_dir_save_visu = path_dir_save_visu

        self.keys_stats = [
            "len_words",
            "special_characters_ratio",
            "stopwords_ratio",
            "badwords_ratio",
            "lang_id_score",
            "perplexity_score",
        ]

    def compute_stats(self):
        stats = []
        for i in tqdm(range(self.num_iter)):
            stats_sentence = {key_stats: None for key_stats in self.keys_stats}

            sentence = self.ds[i]["text"]

            len_words = [len(word) for word in sentence.split(" ")]
            stats_sentence["len_words"] = len_words

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

            stats.append(stats_sentence)
        self.stats = stats

    def plot(self):
        pathlib.Path(self.path_dir_save_visu).mkdir(parents=True, exist_ok=True)
        path_dir_save_visu_lang = pathlib.PurePath(
            self.path_dir_save_visu, self.lang_oscar_id
        )
        pathlib.Path(path_dir_save_visu_lang).mkdir(parents=True, exist_ok=True)

        def plot_histogram(data, xlabel, ylabel, title, path_save, n_bins=50):
            fig = plt.figure()
            plt.hist(data, n_bins, density=True)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True)
            plt.savefig(path_save)
            plt.close(fig)

        for key in self.keys_stats:
            data = [sent[key] for sent in self.stats if sent[key]]
            if key == "len_words":
                data = [el for list_ in data for el in list_]
            if len(data) > 0:
                if key == "len_words":
                    quantile = np.quantile(data, 0.99)
                    data = np.array(data)
                    data = data[data < quantile]
                xlabel = key.replace("_", " ")
                ylabel = "probability"
                title = f"Language {self.lang_oscar_id}"
                path_save = pathlib.PurePath(
                    path_dir_save_visu_lang, f"{key}_{self.lang_oscar_id}.png"
                )
                plot_histogram(data, xlabel, ylabel, title, path_save)

    def get_examples_above_cutoff(self, key, cutoff):
        if key not in self.keys_stats:
            raise ValueError("Specified key unknown.")
        if key == "len_words":
            examples = []
            for i in range(self.num_iter):
                words = [
                    word for word in self.ds[i]["text"].split(" ") if len(word) > cutoff
                ]
                examples += words
        else:
            examples = [
                self.ds[i]["text"]
                for i in range(self.num_iter)
                if self.stats[i][key] and (self.stats[i][key] > cutoff)
            ]
        return examples


if __name__ == "__main__":
    lang_oscar_id = "af"
    path_fasttext_model = "/tmp/lid.176.bin"
    path_kenlm_model = "ac_dc/af.arpa.bin"
    path_sentence_piece_model = "ac_dc/af.sp.model"
    num_iter = 10000
    path_dir_save_visu = "../Oscar_filtering_visualization/"

    filtering_visualization = FilteringVisualization(
        lang_oscar_id,
        path_fasttext_model,
        path_kenlm_model,
        path_sentence_piece_model,
        num_iter,
        path_dir_save_visu,
    )
    filtering_visualization.compute_stats()
    filtering_visualization.plot()
    print(filtering_visualization.get_examples_above_cutoff("len_words", 25))
    print(
        filtering_visualization.get_examples_above_cutoff(
            "special_characters_ratio", 0.5
        )
    )

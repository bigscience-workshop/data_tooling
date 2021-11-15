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
            lang_oscar_id, path_kenlm_model
        )
        self.param = LoadParameters.load_parameters(lang_oscar_id)

        self.ds = load_dataset(
            "oscar",
            f"unshuffled_deduplicated_{lang_oscar_id}",
            split="train",
            streaming=True,
        ).shuffle(buffer_size=num_iter, seed=42)
        self.num_iter = num_iter
        self.path_dir_save_visu = path_dir_save_visu

        self.stats = {
            "len_words": [],
            "special_characters_ratios": [],
            "stopwords_ratios": [],
            "badwords_ratios": [],
            "lang_id_scores": [],
            "perplexity_scores": [],
        }

    def compute_stats(self):
        dataset = iter(self.ds)
        for i in tqdm(range(self.num_iter)):
            sentence = next(dataset)["sentence"]

            len_words = [len(word) for word in sentence.split(" ")]
            self.stats["len_words"] += len_words

            special_characters_ratio = Filtering.compute_special_characters_ratio(
                sentence, self.param["special_characters"]
            )
            self.stats["special_characters_ratios"].append(special_characters_ratio)

            if self.stopwords:
                stopwords_ratio = Filtering.compute_stopwords_ratio(
                    sentence, self.param["strip_characters"], self.stopwords
                )
                self.stats["stopwords_ratios"].append(stopwords_ratio)

            if self.badwords:
                badwords_ratio = Filtering.compute_badwords_ratio(
                    sentence, self.param["strip_characters"], self.badwords
                )
                self.stats["badwords_ratios"].append(badwords_ratio)

            if self.model_lang_id:
                _, lang_id_score = Filtering.compute_lang_id_pred_score(
                    sentence, self.param["strip_characters"], self.model_lang_id
                )
                self.stats["lang_id_scores"].append(lang_id_score)

            if self.kenlm_model:
                perplexity_score = Filtering.compute_perplexity_score(
                    sentence, self.kenlm_model
                )
                self.stats["perplexity_scores"].append(perplexity_score)

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

        for key in self.stats:
            if len(self.stats[key]) > 0:
                data = self.stats[key]
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


if __name__ == "__main__":
    lang_oscar_id = "af"
    path_fasttext_model = "/tmp/lid.176.bin"
    path_kenlm_model = "ac_dc/af.arpa.bin"
    num_iter = 10000
    path_dir_save_visu = "../Oscar_filtering_visualization/"

    filtering_visualization = FilteringVisualization(
        lang_oscar_id,
        path_fasttext_model,
        path_kenlm_model,
        num_iter,
        path_dir_save_visu,
    )
    filtering_visualization.compute_stats()
    filtering_visualization.plot()

# Run with: streamlit run visualization.py

import streamlit as st

import json
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, path_data, lang, num_docs, num_docs_for_words):
        self.path_data = path_data
        self.lang = lang
        self.num_docs = num_docs
        self.num_docs_for_words = num_docs_for_words

    def open_data(self):
        with open(self.path_data) as json_file:
            data = json.load(json_file)

        self.num_docs = min(self.num_docs, len(data))
        self.num_docs_for_words = min(self.num_docs_for_words, len(data))

        sentences = [doc["text"].split(" ") for doc in data[:num_docs_for_words]]
        words = [word for sentence in sentences for word in sentence]
        words_data = [{"len_word": len(word), "word": word} for word in words]
        self.words_data = pd.DataFrame(words_data)

        data = data[:num_docs]
        self.data = pd.DataFrame(data)

    def set_title(self):
        st.title(f"{self.num_docs} {self.lang} documents from Oscar with their stats.")

    def filtering_of_docs(self):
        st.sidebar.subheader("Parameters of the filtering on documents")

        def set_sliders(data):
            columns = list(data)
            keys = []

            if "number_words" in columns:
                max_nb_words = int(np.max(data["number_words"])) + 1
                cutoff_number_words = st.sidebar.slider(
                    "Min cutoff number words", 0, max_nb_words, 0
                )
                keys.append(("number_words", cutoff_number_words, False))

            if "special_characters_ratio" in columns:
                cutoff_special_characters_ratio = st.sidebar.slider(
                    "Max cutoff special characters ratio", 0.0, 1.0, 1.0, step=0.01
                )
                keys.append(("special_characters_ratio", cutoff_special_characters_ratio, True))

            if "stopwords_ratio" in columns:
                cutoff_stopwords_ratio = st.sidebar.slider(
                    "Min cutoff stopwords ratio", 0.0, 1.0, 0.0, step=0.01
                )
                keys.append(("stopwords_ratio", cutoff_stopwords_ratio, False))

            if "badwords_ratio" in columns:
                cutoff_badwords_ratio = st.sidebar.slider(
                    "Max cutoff badwords ratio", 0.0, 1.0, 1.0, step=0.001
                )
                keys.append(("badwords_ratio", cutoff_badwords_ratio, True))

            if "lang_id_score" in columns:
                cutoff_lang_id_score = st.sidebar.slider(
                    "Min cutoff lang id score", 0.0, 1.0, 0.0, step=0.01
                )
                keys.append(("lang_id_score", cutoff_lang_id_score, False))

            if "perplexity_score" in columns:
                max_pp = int(np.max(data["perplexity_score"])) + 1
                cutoff_perplexity_score = st.sidebar.slider(
                    "Perplexity cutoff perplexity score", 0, max_pp, max_pp
                )
                keys.append(("perplexity_score", cutoff_perplexity_score, True))

            return keys

        self.keys = set_sliders(self.data)

        cond = [
            (self.data[key] <= cutoff) if max_cutoff else (self.data[key] >= cutoff)
            for key, cutoff, max_cutoff in self.keys
        ]
        cond = np.all(cond, axis=0)

        st.header("Filtering on documents")

        self.data_not_keep = self.data.loc[np.invert(cond)]
        st.subheader(f"Discarded documents: {len(self.data_not_keep)} docs ({len(self.data_not_keep) / num_docs * 100:.2f}%)")
        st.markdown("Click on a column to sort by it, place the cursor on the text to display it.")
        st.dataframe(self.data_not_keep)

        self.data_keep = self.data.loc[cond]
        st.subheader(f"Retained documents: {len(self.data_keep)} docs ({len(self.data_keep) / num_docs * 100:.2f}%)")
        st.markdown("Click on a column to sort by it, place the cursor on the text to display it.")
        st.dataframe(self.data_keep)

    def filtering_of_words(self):
        st.sidebar.subheader("Parameter of the filtering for words")

        max_len_word = int(np.max(self.words_data["len_word"])) + 1
        cutoff_word = st.sidebar.slider("Max cutoff length word", 0, max_len_word, max_len_word)
        cond_words = self.words_data["len_word"] <= cutoff_word

        st.header("Filtering on words")

        st.markdown(
            (
                f"Since the number of words is way larger than the number of documents, "
                f"we consider in this section words for the first {num_docs_for_words} documents only."
            )
        )

        words_not_keep = self.words_data.loc[np.invert(cond_words)]
        st.subheader(f"Discarded words: {len(words_not_keep)} words ({len(words_not_keep) / len(self.words_data) * 100:.2f}%)")
        st.markdown("Click on a column to sort by it, place the cursor on the text to display it.")
        st.dataframe(words_not_keep)

        words_keep = self.words_data.loc[cond_words]
        st.subheader(f"Retained words: {len(words_keep)} words ({len(words_keep) / len(self.words_data) * 100:.2f}%)")
        st.markdown("Click on a column to sort by it, place the cursor on the text to display it.")
        st.dataframe(words_keep)

    def plot_distributions_filtering_parameters(self):
        st.header("Distributions of the filtering parameters")

        display_distributions = st.checkbox("Display distributions")

        if display_distributions:

            def plot_hist(dataframe, key, num_bins=50):
                st.subheader(" ".join(key.split("_")))
                hist_values = dataframe[key].values
                max_range = np.max(hist_values)
                hist_values = np.histogram(hist_values, bins=num_bins, range=(0, max_range))[0]
                st.bar_chart(hist_values)
                st.markdown(f"Each bin is of size: {max_range/num_bins}.")

            for key, _, _ in self.keys:
                plot_hist(self.data, key)

    def plot_zipf_laws(self):
        st.header("Zipf's Laws")

        display_zipf_laws = st.checkbox("Display Zipf's Laws")

        if display_zipf_laws:

            def get_frequency_words(data):
                freq_words = {}
                for index, row in data.iterrows():
                    for word in row["text"].split(" "):
                        if word in freq_words:
                            freq_words[word] += 1
                        else:
                            freq_words[word] = 1
                freq_words = np.array(list(freq_words.values()))
                freq_words = -np.sort(-freq_words)
                return freq_words

            freq_words_data = get_frequency_words(self.data)
            freq_words_data_keep = get_frequency_words(self.data_keep)
            freq_words_data_not_keep = get_frequency_words(self.data_not_keep)

            fig, ax = plt.subplots()
            ax.loglog(freq_words_data)
            ax.loglog(freq_words_data_keep)
            ax.loglog(freq_words_data_not_keep)
            ax.set_title("Zipf's Law")
            ax.set_xlabel("$i$-th most frequent word")
            ax.set_ylabel("frequency in the documents")
            ax.legend(["All docs", "Retained docs", "Discarded docs"])
            st.pyplot(fig)

            st.markdown(
                "If less than three curves are displayed, it means that there are overlaps."
            )

    def download_data(self):
        st.header("Download data")

        with open(self.path_data) as json_file:
            btn = st.download_button(
                label="Download data as json",
                data=json_file,
                file_name="data.json",
            )

    def visualization(self):
        self.open_data()
        self.set_title()
        self.filtering_of_docs()
        self.filtering_of_words()
        self.plot_distributions_filtering_parameters()
        self.plot_zipf_laws()
        self.download_data()


path_data = "./ac_dc/visualization/en_examples_with_stats.json"
lang = "English"
num_docs = 5000
num_docs_for_words = 500

visualization = Visualization(path_data, lang, num_docs, num_docs_for_words)
visualization.visualization()

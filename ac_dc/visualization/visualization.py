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
        words = [{"len_word": len(word), "word": word} for word in words]
        self.words = pd.DataFrame(words)

        docs = data[:num_docs]
        self.docs = pd.DataFrame(docs)

    def set_title(self):
        st.title(f"{self.num_docs} {self.lang} documents from Oscar with their stats.")

    def filtering_of_docs(self):
        st.sidebar.subheader("Parameters of the filtering on documents")

        def set_sliders(docs):
            columns = list(docs)
            keys = []

            if "number_words" in columns:
                max_nb_words = int(np.max(docs["number_words"])) + 1
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
                max_pp = int(np.max(docs["perplexity_score"])) + 1
                cutoff_perplexity_score = st.sidebar.slider(
                    "Perplexity cutoff perplexity score", 0, max_pp, max_pp
                )
                keys.append(("perplexity_score", cutoff_perplexity_score, True))

            return keys

        self.keys = set_sliders(self.docs)

        cond = [
            (self.docs[key] <= cutoff) if max_cutoff else (self.docs[key] >= cutoff)
            for key, cutoff, max_cutoff in self.keys
        ]
        cond = np.all(cond, axis=0)

        st.header("Filtering on documents")

        self.discarded_docs = self.docs.loc[np.invert(cond)]
        st.subheader(f"Discarded documents: {len(self.discarded_docs)} docs ({len(self.discarded_docs) / num_docs * 100:.2f}%)")
        st.markdown("Click on a column to sort by it, place the cursor on the text to display it.")
        st.dataframe(self.discarded_docs)

        self.retained_docs = self.docs.loc[cond]
        st.subheader(f"Retained documents: {len(self.retained_docs)} docs ({len(self.retained_docs) / num_docs * 100:.2f}%)")
        st.markdown("Click on a column to sort by it, place the cursor on the text to display it.")
        st.dataframe(self.retained_docs)

    def filtering_of_words(self):
        st.sidebar.subheader("Parameter of the filtering for words")

        max_len_word = int(np.max(self.words["len_word"])) + 1
        cutoff_word = st.sidebar.slider("Max cutoff length word", 0, max_len_word, max_len_word)
        cond_words = self.words["len_word"] <= cutoff_word

        st.header("Filtering on words")

        st.markdown(
            (
                f"Since the number of words is way larger than the number of documents, "
                f"we consider in this section words for the first {num_docs_for_words} documents only."
            )
        )

        discarded_words = self.words.loc[np.invert(cond_words)]
        st.subheader(f"Discarded words: {len(discarded_words)} words ({len(discarded_words) / len(self.words) * 100:.2f}%)")
        st.markdown("Click on a column to sort by it, place the cursor on the text to display it.")
        st.dataframe(discarded_words)

        retained_words = self.words.loc[cond_words]
        st.subheader(f"Retained words: {len(retained_words)} words ({len(retained_words) / len(self.words) * 100:.2f}%)")
        st.markdown("Click on a column to sort by it, place the cursor on the text to display it.")
        st.dataframe(retained_words)

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
                plot_hist(self.docs, key)

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

            freq_words_data = get_frequency_words(self.docs)
            freq_words_data_keep = get_frequency_words(self.retained_docs)
            freq_words_data_not_keep = get_frequency_words(self.discarded_docs)

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

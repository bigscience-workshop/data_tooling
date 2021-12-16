# Run with: streamlit run visualization.py

import streamlit as st

import json
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


class Visualization:
    def __init__(
        self, path_data, lang, num_docs, num_docs_for_words, max_len_text_display
    ):
        self.path_data = path_data
        self.lang = lang
        self.num_docs = num_docs
        self.num_docs_for_words = num_docs_for_words
        self.max_len_text_display = max_len_text_display

    def open_data(self):
        with open(self.path_data) as json_file:
            data = json.load(json_file)

        self.num_docs = min(self.num_docs, len(data))
        self.num_docs_for_words = min(self.num_docs_for_words, len(data))

        words = [doc["words"] for doc in data[: self.num_docs_for_words]]
        words = [word for doc in words for word in doc]
        self.words = pd.DataFrame(words)

        docs = data[: self.num_docs]
        for doc in docs:
            del doc["words"]
            if len(doc["text"]) > self.max_len_text_display:
                doc["text"] = (
                    doc["text"][: self.max_len_text_display]
                    + " [...] [THIS LONG TEXT HAS BEEN TRUNCATED FOR DISPLAY REASONS]"
                )
        self.docs = pd.DataFrame(docs)

    def set_title(self):
        st.title(f"{self.num_docs} {self.lang} documents from Oscar with their stats.")

    def filtering_of_docs(self):
        st.sidebar.subheader("Parameters of the filtering on documents")

        def set_sliders(docs):
            columns = list(docs)
            keys = []
            conds = []

            def get_cond(key, cutoff, max_cutoff):
                if max_cutoff:
                    return self.docs[key] <= cutoff
                return self.docs[key] >= cutoff

            def print_discared_by_cond(cond):
                st.sidebar.caption(
                    f"{(len(cond) - np.sum(1*cond)) / len(cond) * 100:.2f}% of the total is discarded with this filter."
                )
                st.sidebar.caption("---------")

            if "number_words" in columns:
                cutoff_def = "If the number of words of a document is lower than this number, the document is removed."
                max_nb_words = int(np.max(docs["number_words"])) + 1
                cutoff_min_number_words = st.sidebar.slider(
                    cutoff_def, 0, min(max_nb_words, 500), 0
                )
                new_key = ("number_words", cutoff_min_number_words, False)
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                conds.append(cond)
                print_discared_by_cond(cond)

                cutoff_def = "If the number of words of a document is higher than this number, the document is removed."
                cutoff_max_number_words = st.sidebar.slider(
                    cutoff_def, 0, max_nb_words, max_nb_words
                )
                new_key = ("number_words", cutoff_max_number_words, True)
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                conds.append(cond)
                print_discared_by_cond(cond)

            if "special_characters_ratio" in columns:
                cutoff_def = "If the special characters ratio of a document is higher than this number, the document is removed."
                cutoff_special_characters_ratio = st.sidebar.slider(
                    cutoff_def, 0.0, 1.0, 1.0, step=0.01
                )
                new_key = (
                    "special_characters_ratio",
                    cutoff_special_characters_ratio,
                    True,
                )
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                conds.append(cond)
                print_discared_by_cond(cond)

            if "stopwords_ratio" in columns:
                cutoff_def = "If the stop words ratio of a document is lower than this number, the document is removed."
                cutoff_stopwords_ratio = st.sidebar.slider(
                    cutoff_def, 0.0, 1.0, 0.0, step=0.01
                )
                new_key = ("stopwords_ratio", cutoff_stopwords_ratio, False)
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                conds.append(cond)
                print_discared_by_cond(cond)

            if "badwords_ratio" in columns:
                cutoff_def = "If the bad words ratio of a document is higher than this number, the document is removed."
                cutoff_badwords_ratio = st.sidebar.slider(
                    cutoff_def, 0.0, 1.0, 1.0, step=0.01
                )
                new_key = ("badwords_ratio", cutoff_badwords_ratio, True)
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                conds.append(cond)
                print_discared_by_cond(cond)

            if "lang_id_score" in columns:
                cutoff_def = "If the confidence score for the language identification prediction of a document is lower than this number, the document is removed."
                cutoff_lang_id_score = st.sidebar.slider(
                    cutoff_def, 0.0, 1.0, 0.0, step=0.01
                )
                new_key = ("lang_id_score", cutoff_lang_id_score, False)
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                conds.append(cond)
                print_discared_by_cond(cond)

            if "perplexity_score" in columns:
                cutoff_def = "If the perplexity score of a document is higher than this number, the document is removed."
                max_pp = int(np.max(docs["perplexity_score"])) + 1
                cutoff_perplexity_score = st.sidebar.slider(
                    cutoff_def, 0, max_pp, max_pp
                )
                new_key = ("perplexity_score", cutoff_perplexity_score, True)
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                conds.append(cond)
                print_discared_by_cond(cond)

            return keys, conds

        self.keys, conds = set_sliders(self.docs)

        conds = np.all(conds, axis=0)

        st.header("Filtering on documents")

        self.discarded_docs = self.docs.loc[np.invert(conds)]
        st.subheader(
            f"Discarded documents: {len(self.discarded_docs)} docs ({len(self.discarded_docs) / self.num_docs * 100:.2f}%)"
        )
        st.markdown(
            "Click on a column to sort by it, place the cursor on the text to display it."
        )
        st.dataframe(self.discarded_docs)

        self.retained_docs = self.docs.loc[conds]
        st.subheader(
            f"Retained documents: {len(self.retained_docs)} docs ({len(self.retained_docs) / self.num_docs * 100:.2f}%)"
        )
        st.markdown(
            "Click on a column to sort by it, place the cursor on the text to display it."
        )
        st.dataframe(self.retained_docs)

    def filtering_of_words(self):
        st.sidebar.subheader("Parameter of the filtering on words")

        cutoff_def = (
            "If the length of a word is higher than this number, the word is removed."
        )
        max_len_word = min(int(np.max(self.words["len_word"])) + 1, 200)
        cutoff_word = st.sidebar.slider(cutoff_def, 0, max_len_word, max_len_word)

        incorrect_substrings = st.sidebar.checkbox(
            "Remove words with incorrect substrings."
        )

        cond_words = self.words["len_word"] <= cutoff_word
        if incorrect_substrings:
            cond_words = cond_words & np.invert(self.words["incorrect_substring"])

        st.header("Filtering on words")

        st.markdown(
            f"Since the number of words is way larger than the number of documents, "
            f"we consider in this section words for the first {self.num_docs_for_words} documents only."
        )

        discarded_words = self.words.loc[np.invert(cond_words)]
        st.subheader(
            f"Discarded words: {len(discarded_words)} words ({len(discarded_words) / len(self.words) * 100:.2f}%)"
        )
        st.markdown(
            "Click on a column to sort by it, place the cursor on the text to display it."
        )
        st.dataframe(discarded_words)

        retained_words = self.words.loc[cond_words]
        st.subheader(
            f"Retained words: {len(retained_words)} words ({len(retained_words) / len(self.words) * 100:.2f}%)"
        )
        st.markdown(
            "Click on a column to sort by it, place the cursor on the text to display it."
        )
        st.dataframe(retained_words)

    def plot_distributions_filtering_parameters(self):
        st.header("Distributions of the filtering parameters")

        display_distributions = st.checkbox("Display distributions")

        if display_distributions:

            def plot_hist(dataframe, key, num_bins=50):
                st.subheader(" ".join(key.split("_")))
                hist_values = dataframe[key].values
                max_range = np.max(hist_values)
                hist_values = np.histogram(
                    hist_values, bins=num_bins, range=(0, max_range)
                )[0]
                st.bar_chart(hist_values)
                st.markdown(f"Each bin is of size: {max_range/num_bins}.")

            for key in list({el[0]: None for el in self.keys}):
                plot_hist(self.docs, key)

            plot_hist(self.words, "len_word")

    def plot_zipf_law(self):
        st.header("Zipf's Law")

        display_zipf_law = st.checkbox("Display Zipf's Law")

        if display_zipf_law:

            freq_words = {}
            for _, row in self.words.iterrows():
                freq_words[row["word"]] = freq_words.get(row["word"], 0) + 1
            freq_words = np.array(list(freq_words.values()))
            freq_words = -np.sort(-freq_words)

            fig, ax = plt.subplots()
            ax.loglog(freq_words)
            ax.set_title("Zipf's Law")
            ax.set_xlabel("$i$-th most frequent word")
            ax.set_ylabel("frequency in the documents")
            st.pyplot(fig)

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
        self.plot_zipf_law()
        self.download_data()


path_data = "./ac_dc/visualization/en_examples_with_stats.json"
lang = "English"
num_docs = 15000
num_docs_for_words = 1500
max_len_text_display = 10000

visualization = Visualization(
    path_data, lang, num_docs, num_docs_for_words, max_len_text_display
)
visualization.visualization()

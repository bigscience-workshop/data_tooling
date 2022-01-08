# Run with: streamlit run visualization.py

import streamlit as st

import os

import base64
import json
import pandas as pd
pd.options.mode.chained_assignment = None

import numpy as np

import matplotlib.pyplot as plt


class Visualization:
    def __init__(
        self,
        path_instructions,
        path_data,
        lang,
        num_docs,
        num_docs_for_words,
        max_len_text_display,
    ):
        self.path_instructions = path_instructions
        self.path_data = path_data
        self.lang = lang
        self.num_docs = num_docs
        self.num_docs_for_words = num_docs_for_words
        self.max_len_text_display = max_len_text_display

    def preamble(self):
        st.markdown(
            "Before diving into this demo, you might want to take a look at how the filtering pipeline looks like in more detail."
        )

        def get_binary_file_downloader_html(bin_file, file_label="File"):
            with open(bin_file, "rb") as f:
                data = f.read()
            bin_str = base64.b64encode(data).decode()
            href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
            return href

        st.markdown(
            get_binary_file_downloader_html(
                self.path_instructions,
                "Download the explanation of the filtering pipeline as pdf",
            ),
            unsafe_allow_html=True,
        )

    def open_data(self):
        with open(self.path_data) as json_file:
            data = json.load(json_file)

        self.num_docs = min(self.num_docs, len(data))
        self.num_docs_for_words = min(self.num_docs_for_words, len(data))

        if "words" in data[0]:
            words = [doc["words"] for doc in data[: self.num_docs_for_words]]
            words = [word for doc in words for word in doc]
            self.words = pd.DataFrame(words)
        else:
            self.words = None

        docs = data[: self.num_docs]
        for doc in docs:
            if not (self.words is None):
                del doc["words"]
            if len(doc["text"]) > self.max_len_text_display:
                doc["text"] = (
                    doc["text"][: self.max_len_text_display]
                    + " [...] [THIS LONG TEXT HAS BEEN TRUNCATED FOR DISPLAY REASONS]"
                )
        self.docs_checkpoint = pd.DataFrame(docs)
        self.docs = self.docs_checkpoint

    def set_title(self):
        st.title(f"{self.num_docs} {self.lang} documents with their stats.")

    def filtering_of_docs(self):
        st.sidebar.subheader("Parameters of the filtering on documents")

        def set_sliders():
            columns = list(self.docs)
            keys = []
            conds = {}

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
                max_nb_words = int(np.max(self.docs["number_words"])) + 1
                cutoff_min_number_words = st.sidebar.slider(
                    cutoff_def, 0, min(max_nb_words, 500), 0
                )
                new_key = ("number_words", cutoff_min_number_words, False)
                keys.append(new_key)
                cond_1 = get_cond(new_key[0], new_key[1], new_key[2])
                print_discared_by_cond(cond_1)

                cutoff_def = "If the number of words of a document is higher than this number, the document is removed."
                cutoff_max_number_words = st.sidebar.slider(
                    cutoff_def, 0, max_nb_words, max_nb_words
                )
                new_key = ("number_words", cutoff_max_number_words, True)
                keys.append(new_key)
                cond_2 = get_cond(new_key[0], new_key[1], new_key[2])
                print_discared_by_cond(cond_2)

                conds["number_words"] = [cond_1, cond_2]

            if "repetitions_ratio" in columns:
                val_repetitions_lengths = list(
                    self.docs["repetitions_ratio"].iloc[0].keys()
                )
                default_index = (
                    val_repetitions_lengths.index("10")
                    if "10" in val_repetitions_lengths
                    else 0
                )
                label_selectbox = (
                    "Length of the repetitions (that will determine the repetitions ratio). "
                    "Choosing a higher or lower number does not mean that the filtering "
                    "is stronger or weaker. Be careful, choosing a low number (below 5 for languages like English) "
                    "tends to associate a high repetitions ratio to very long documents (like book chapters), but with "
                    "few or no repetitions, simply because their length gives them more diversity, and we do "
                    "not want to discard such documents."
                )
                repetitions_length = st.sidebar.selectbox(
                    label=label_selectbox,
                    options=val_repetitions_lengths,
                    index=default_index,
                )
                self.docs = self.docs_checkpoint
                for i in range(len(self.docs["repetitions_ratio"])):
                    self.docs["repetitions_ratio"].iloc[i] = self.docs["repetitions_ratio"].iloc[i][repetitions_length]

                cutoff_def = "If the repetitions ratio of a document is higher than this number, the document is removed."
                cutoff_repetitions_ratio = st.sidebar.slider(
                    cutoff_def, 0.0, 1.0, 1.0, step=0.01
                )
                new_key = (
                    "repetitions_ratio",
                    cutoff_repetitions_ratio,
                    True,
                )
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                print_discared_by_cond(cond)
                conds["repetitions_ratio"] = [cond]

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
                print_discared_by_cond(cond)
                conds["special_characters_ratio"] = [cond]

            if "stopwords_ratio" in columns:
                cutoff_def = "If the stop words ratio of a document is lower than this number, the document is removed."
                cutoff_stopwords_ratio = st.sidebar.slider(
                    cutoff_def, 0.0, 1.0, 0.0, step=0.01
                )
                new_key = ("stopwords_ratio", cutoff_stopwords_ratio, False)
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                print_discared_by_cond(cond)
                conds["stopwords_ratio"] = [cond]

            if "badwords_ratio" in columns:
                cutoff_def = "If the bad words ratio of a document is higher than this number, the document is removed."
                cutoff_badwords_ratio = st.sidebar.slider(
                    cutoff_def, 0.0, 1.0, 1.0, step=0.01
                )
                new_key = ("badwords_ratio", cutoff_badwords_ratio, True)
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                print_discared_by_cond(cond)
                conds["badwords_ratio"] = [cond]

            if "lang_id_score" in columns:
                cutoff_def = "If the confidence score for the language identification prediction of a document is lower than this number, the document is removed."
                cutoff_lang_id_score = st.sidebar.slider(
                    cutoff_def, 0.0, 1.0, 0.0, step=0.01
                )
                new_key = ("lang_id_score", cutoff_lang_id_score, False)
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                print_discared_by_cond(cond)
                conds["lang_id_score"] = [cond]

            if "perplexity_score" in columns:
                cutoff_def = "If the perplexity score of a document is higher than this number, the document is removed."
                max_pp = int(np.max(self.docs["perplexity_score"])) + 1
                cutoff_perplexity_score = st.sidebar.slider(
                    cutoff_def, 0, max_pp, max_pp
                )
                new_key = ("perplexity_score", cutoff_perplexity_score, True)
                keys.append(new_key)
                cond = get_cond(new_key[0], new_key[1], new_key[2])
                print_discared_by_cond(cond)
                conds["perplexity_score"] = [cond]

            return keys, conds

        self.keys, conds = set_sliders()

        all_conds = [subcond for cond in list(conds.values()) for subcond in cond]
        all_conds = np.all(all_conds, axis=0)

        st.header("Filtering on documents")

        def display_dataset(cond, description):
            displayed_docs = self.docs.loc[cond]
            st.subheader(
                f"{description}: {len(displayed_docs)} docs ({len(displayed_docs) / self.num_docs * 100:.2f}%)"
            )
            st.markdown(
                "Click on a column to sort by it, place the cursor on the text to display it."
            )
            st.dataframe(displayed_docs)

        display_dataset(np.invert(all_conds), "Discarded documents")

        # st.subheader("Display discarded documents by filter")
        display_discarded_documents_by_filter = st.checkbox(
            "Display discarded documents by filter"
        )

        if display_discarded_documents_by_filter:
            columns = list(self.docs)

            if "number_words" in columns:
                cond_filter = np.invert(np.all(conds["number_words"], axis=0))
                display_dataset(
                    cond_filter,
                    "Discarded documents for the filter on the number of words",
                )

            if "repetitions_ratio" in columns:
                cond_filter = np.invert(np.all(conds["repetitions_ratio"], axis=0))
                display_dataset(
                    cond_filter,
                    "Discarded documents for the filter on the repetitions ratio",
                )

            if "special_characters_ratio" in columns:
                cond_filter = np.invert(
                    np.all(conds["special_characters_ratio"], axis=0)
                )
                display_dataset(
                    cond_filter,
                    "Discarded documents for the filter on the special characters ratio",
                )

            if "stopwords_ratio" in columns:
                cond_filter = np.invert(np.all(conds["stopwords_ratio"], axis=0))
                display_dataset(
                    cond_filter,
                    "Discarded documents for the filter on the stop words ratio",
                )

            if "badwords_ratio" in columns:
                cond_filter = np.invert(np.all(conds["badwords_ratio"], axis=0))
                display_dataset(
                    cond_filter,
                    "Discarded documents for the filter on the bad words ratio",
                )

            if "lang_id_score" in columns:
                cond_filter = np.invert(np.all(conds["lang_id_score"], axis=0))
                display_dataset(
                    cond_filter,
                    "Discarded documents for the filter on the language identification confidence score",
                )

            if "perplexity_score" in columns:
                cond_filter = np.invert(np.all(conds["perplexity_score"], axis=0))
                display_dataset(
                    cond_filter,
                    "Discarded documents for the filter on the perplexity score",
                )

        display_dataset(all_conds, "Retained documents")

    def filtering_of_words(self):
        if not (self.words is None):
            st.sidebar.subheader("Parameter of the filtering on words")

            cutoff_def = "If the length of a word is higher than this number, the word is removed."
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

            if not (self.words is None):
                plot_hist(self.words, "len_word")

    def plot_zipf_law(self):
        if not (self.words is None):
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
        self.preamble()
        self.open_data()
        self.set_title()
        self.filtering_of_docs()
        self.filtering_of_words()
        self.plot_distributions_filtering_parameters()
        self.plot_zipf_law()
        self.download_data()


path_instructions = "./ac_dc/explanation_filtering_pipeline.pdf"
path_data = "./ac_dc/visualization/en_examples_with_stats.json"
lang = "English"
num_docs = 15000
num_docs_for_words = 1500
max_len_text_display = 10000

visualization = Visualization(
    path_instructions,
    path_data,
    lang,
    num_docs,
    num_docs_for_words,
    max_len_text_display,
)
visualization.visualization()

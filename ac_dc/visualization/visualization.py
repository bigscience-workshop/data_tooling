# Run with: streamlit run visualization.py

import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualization(path_data, lang, num_docs, num_docs_for_words):

    with open(path_data) as json_file:
        data = json.load(json_file)

    num_docs = min(num_docs, len(data))

    st.title(f"{num_docs} {lang} documents from Oscar with their stats.")

    sentences = [doc["text"].split(" ") for doc in data[:num_docs_for_words]]
    words = [word for sentence in sentences for word in sentence]
    words_data = [{"len_word": len(word), "word": word} for word in words]
    words_data = pd.DataFrame(words_data)

    data = data[:num_docs]
    data = pd.DataFrame(data)

    columns = list(data)
    keys = []

    st.header("Parameters of the filtering")

    if "number_words" in columns:
        max_nb_words = int(np.max(data["number_words"])) + 1
        cutoff_number_words = st.slider(
            "Min cutoff number words", 0, max_nb_words, 0
        )
        keys.append(("number_words", cutoff_number_words, False))

    if "special_characters_ratio" in columns:
        cutoff_special_characters_ratio = st.slider(
            "Max cutoff special characters ratio", 0.0, 1.0, 1.0, step=0.01
        )
        keys.append(("special_characters_ratio", cutoff_special_characters_ratio, True))

    if "stopwords_ratio" in columns:
        cutoff_stopwords_ratio = st.slider(
            "Min cutoff stopwords ratio", 0.0, 1.0, 0.0, step=0.01
        )
        keys.append(("stopwords_ratio", cutoff_stopwords_ratio, False))

    if "badwords_ratio" in columns:
        cutoff_badwords_ratio = st.slider(
            "Max cutoff badwords ratio", 0.0, 1.0, 1.0, step=0.001
        )
        keys.append(("badwords_ratio", cutoff_badwords_ratio, True))

    if "lang_id_score" in columns:
        cutoff_lang_id_score = st.slider(
            "Min cutoff lang id score", 0.0, 1.0, 0.0, step=0.01
        )
        keys.append(("lang_id_score", cutoff_lang_id_score, False))

    if "perplexity_score" in columns:
        max_pp = int(np.max(data["perplexity_score"])) + 1
        cutoff_perplexity_score = st.slider(
            "Perplexity cutoff perplexity score", 0, max_pp, max_pp
        )
        keys.append(("perplexity_score", cutoff_perplexity_score, True))

    cond = [
        (data[key] <= cutoff) if max_cutoff else (data[key] >= cutoff)
        for key, cutoff, max_cutoff in keys
    ]
    cond = np.all(cond, axis=0)

    data_keep = data.loc[cond]
    st.header(f"Data that we keep: {len(data_keep)} docs")
    st.markdown("Click on a column to sort by it.")
    st.markdown("Place the cursor on the text to display it.")
    st.dataframe(data_keep)

    data_not_keep = data.loc[np.invert(cond)]
    st.header(f"Data that is thrown away: {len(data_not_keep)} docs")
    st.markdown("Click on a column to sort by it.")
    st.markdown("Place the cursor on the text to display it.")
    st.dataframe(data_not_keep)

    def plot_hist(dataframe, key, num_bins=50):
        st.header(" ".join(key.split("_")))
        hist_values = dataframe[key].values
        max_range = np.max(hist_values)
        hist_values = np.histogram(hist_values, bins=num_bins, range=(0, max_range))[0]
        st.bar_chart(hist_values)
        st.markdown(f"Each bin is of size: {max_range/num_bins}.")

    for key, _, _ in keys:
        plot_hist(data, key)

    st.header("Zipf's Law")

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

    freq_words_data = get_frequency_words(data)
    freq_words_data_keep = get_frequency_words(data_keep)
    freq_words_data_not_keep = get_frequency_words(data_not_keep)

    fig, ax = plt.subplots()
    ax.loglog(freq_words_data)
    ax.loglog(freq_words_data_keep)
    ax.loglog(freq_words_data_not_keep)
    ax.set_title("Zipf's Law")
    ax.set_xlabel("$i$-th most frequent word")
    ax.set_ylabel("frequency in the documents")
    ax.legend(["All data", "Data that we keep", "Data that is thrown away"])
    st.pyplot(fig)

    st.markdown(
        "If less than three curves are displayed, it means that there are overlaps."
    )

    st.header("Parameter of the filtering for words")
    max_len_word = int(np.max(words_data["len_word"])) + 1
    cutoff_word = st.slider("Max cutoff length word", 0, max_len_word, max_len_word)
    cond_words = words_data["len_word"] <= cutoff_word

    words_keep = words_data.loc[cond_words]
    st.header(f"Words that we keep (for {num_docs_for_words} documents)")
    st.markdown("Click on a column to sort by it.")
    st.markdown("Place the cursor on the text to display it.")
    st.dataframe(words_keep)

    words_not_keep = words_data.loc[np.invert(cond_words)]
    st.header(f"Words that are thrown away (for {num_docs_for_words} documents)")
    st.markdown("Click on a column to sort by it.")
    st.markdown("Place the cursor on the text to display it.")
    st.dataframe(words_not_keep)

    plot_hist(words_data, "len_word")

    st.header("Download data")

    with open(path_data) as json_file:
        btn = st.download_button(
            label="Download data as json",
            data=json_file,
            file_name="data.json",
        )


path_data = "./ac_dc/visualization/en_examples_with_stats.json"
lang = "English"
num_docs = 5000
num_docs_for_words = 500

visualization(path_data, lang, num_docs, num_docs_for_words)

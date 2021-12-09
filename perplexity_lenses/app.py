import logging
from functools import partial

import streamlit as st
from embedding_lenses.data import uploaded_file_to_dataframe
from embedding_lenses.dimensionality_reduction import (
    get_tsne_embeddings,
    get_umap_embeddings,
)
from embedding_lenses.embedding import load_model

from perplexity_lenses import REGISTRY_DATASET
from perplexity_lenses.data import (
    documents_df_to_sentences_df,
    hub_dataset_to_dataframe,
)
from perplexity_lenses.engine import (
    DIMENSIONALITY_REDUCTION_ALGORITHMS,
    DOCUMENT_TYPES,
    EMBEDDING_MODELS,
    LANGUAGES,
    PERPLEXITY_MODELS,
    SEED,
    generate_plot,
)
from perplexity_lenses.perplexity import KenlmModel
from perplexity_lenses.visualization import draw_histogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.title("Perplexity Lenses")
st.write("Visualize text embeddings in 2D using colors to represent perplexity values.")
uploaded_file = st.file_uploader("Choose an csv/tsv file...", type=["csv", "tsv"])
st.write(
    "Alternatively, select a dataset from the [hub](https://huggingface.co/datasets)"
)
col1, col2, col3 = st.columns(3)
with col1:
    hub_dataset = st.text_input("Dataset name", "mc4")
with col2:
    hub_dataset_config = st.text_input("Dataset configuration", "es")
with col3:
    hub_dataset_split = st.text_input("Dataset split", "train")

col4, col5 = st.columns(2)
with col4:
    text_column = st.text_input("Text field name", "text")
with col5:
    language = st.selectbox("Language", LANGUAGES, 12)

col6, col7 = st.columns(2)
with col6:
    doc_type = st.selectbox("Document type", DOCUMENT_TYPES, 1)
with col7:
    sample = st.number_input("Maximum number of documents to use", 1, 100000, 1000)
perplexity_model = st.selectbox(
    "Dataset on which the perplexity model was trained on", PERPLEXITY_MODELS, 0
).lower()

dimensionality_reduction = st.selectbox(
    "Dimensionality Reduction algorithm", DIMENSIONALITY_REDUCTION_ALGORITHMS, 0
)
model_name = st.selectbox("Sentence embedding model", EMBEDDING_MODELS, 0)

advanced_options = st.checkbox(
    "Advanced options (do not modify if using default KenLM models).", value=False
)
lower_case = True
remove_accents = True
normalize_numbers = True
punctuation = 1
if advanced_options:
    lower_case = st.checkbox(
        "Lower case text for KenLM preprocessing (from cc_net)", value=False
    )
    remove_accents = st.checkbox(
        "Remove accents for KenLM preprocessing (from cc_net)", value=False
    )
    normalize_numbers = st.checkbox(
        "Replace numbers with zeros KenLM preprocessing (from cc_net)", value=True
    )
    punctuation = st.number_input(
        "Punctuation mode to use from cc_net KenLM preprocessing", 1, 2, 1
    )

with st.spinner(text="Loading embedding model..."):
    model = load_model(model_name)
dimensionality_reduction_function = (
    partial(get_umap_embeddings, random_state=SEED)
    if dimensionality_reduction == "UMAP"
    else partial(get_tsne_embeddings, random_state=SEED)
)

with st.spinner(text="Loading KenLM model..."):
    kenlm_model = KenlmModel.from_pretrained(
        perplexity_model,
        language,
        lower_case,
        remove_accents,
        normalize_numbers,
        punctuation,
    )

if uploaded_file or hub_dataset:
    with st.spinner("Loading dataset..."):
        if uploaded_file:
            df = uploaded_file_to_dataframe(uploaded_file)
            if doc_type == "Sentence":
                df = documents_df_to_sentences_df(df, text_column, sample, seed=SEED)
            df["perplexity"] = df[text_column].map(kenlm_model.get_perplexity)
        else:
            df = hub_dataset_to_dataframe(
                hub_dataset,
                hub_dataset_config,
                hub_dataset_split,
                sample,
                text_column,
                kenlm_model,
                seed=SEED,
                doc_type=doc_type,
            )

    # Round perplexity
    df["perplexity"] = df["perplexity"].round().astype(int)
    logger.info(
        f"Perplexity range: {df['perplexity'].min()} - {df['perplexity'].max()}"
    )
    plot, plot_registry = generate_plot(
        df,
        text_column,
        "perplexity",
        None,
        dimensionality_reduction_function,
        model,
        seed=SEED,
        context_logger=st.spinner,
        hub_dataset=hub_dataset,
    )
    logger.info("Displaying plots")
    st.bokeh_chart(plot)
    if hub_dataset == REGISTRY_DATASET:
        st.bokeh_chart(plot_registry)
    fig = draw_histogram(df["perplexity"].values)
    st.pyplot(fig)
    logger.info("Done")

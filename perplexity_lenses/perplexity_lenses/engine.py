import logging
import time
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from bokeh.palettes import Turbo256
from bokeh.plotting import Figure
from embedding_lenses.embedding import embed_text
from embedding_lenses.utils import encode_labels
from embedding_lenses.visualization import draw_interactive_scatter_plot
from sentence_transformers import SentenceTransformer

from perplexity_lenses import REGISTRY_DATASET

logger = logging.getLogger(__name__)
EMBEDDING_MODELS = [
    "distiluse-base-multilingual-cased-v1",
    "distiluse-base-multilingual-cased-v2",
    "all-mpnet-base-v2",
    "flax-sentence-embeddings/all_datasets_v3_mpnet-base",
]
DIMENSIONALITY_REDUCTION_ALGORITHMS = ["UMAP", "t-SNE"]
DOCUMENT_TYPES = ["Whole document", "Sentence"]
SEED = 0
LANGUAGES = [
    "af",
    "ar",
    "az",
    "be",
    "bg",
    "bn",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fa",
    "fi",
    "fr",
    "gu",
    "he",
    "hi",
    "hr",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "lt",
    "lv",
    "mk",
    "ml",
    "mn",
    "mr",
    "my",
    "ne",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "uk",
    "zh",
]
PERPLEXITY_MODELS = ["Wikipedia", "OSCAR"]


class ContextLogger:
    def __init__(self, text: str = ""):
        self.text = text
        self.start_time = time.time()

    def __enter__(self):
        logger.info(self.text)

    def __exit__(self, type, value, traceback):
        logger.info(f"Took: {time.time() - self.start_time:.4f} seconds")


def generate_plot(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    sample: Optional[int],
    dimensionality_reduction_function: Callable,
    model: SentenceTransformer,
    seed: int = 0,
    context_logger: Union[st.spinner, ContextLogger] = ContextLogger,
    hub_dataset: str = "",
) -> Tuple[Figure, Optional[Figure]]:
    if text_column not in df.columns:
        raise ValueError(
            f"The specified column name doesn't exist. Columns available: {df.columns.values}"
        )
    if label_column not in df.columns:
        df[label_column] = 0
    df = df.dropna(subset=[text_column, label_column])
    if sample:
        df = df.sample(min(sample, df.shape[0]), random_state=seed)
    with context_logger(text="Embedding text..."):
        embeddings = embed_text(df[text_column].values.tolist(), model)
    logger.info("Encoding labels")
    encoded_labels = encode_labels(df[label_column])
    with context_logger("Reducing dimensionality..."):
        embeddings_2d = dimensionality_reduction_function(embeddings)
    logger.info("Generating figure")
    hover_data = {
        text_column: df[text_column].values,
        label_column: encoded_labels.values,
    }
    # Round perplexity values
    values = df[label_column].values.round().astype(int)
    plot = draw_interactive_scatter_plot(
        hover_data,
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        values,
    )
    # Special case for the registry dataset
    plot_registry = None
    if hub_dataset == REGISTRY_DATASET:
        encoded_labels = encode_labels(df["label"])
        hover_data = {
            text_column: df[text_column].values,
            "label": df["label"].values,
            label_column: df[label_column].values,
        }
        plot_registry = draw_interactive_scatter_plot(
            hover_data,
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            encoded_labels.values,
            palette=Turbo256,
        )
    return plot, plot_registry

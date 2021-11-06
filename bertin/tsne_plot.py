import argparse
import logging
from typing import Any, Optional

import bokeh
import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Cividis256 as Pallete
from bokeh.plotting import figure, output_file, save
from bokeh.transform import factor_cmap
from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SEED = 0


def get_tsne_embeddings(
    embeddings: np.ndarray,
    perplexity: int = 30,
    n_components: int = 2,
    init: str = "pca",
    n_iter: int = 5000,
    random_state: int = SEED,
) -> np.ndarray:
    tsne = TSNE(
        perplexity=perplexity,
        n_components=n_components,
        init=init,
        n_iter=n_iter,
        random_state=random_state,
    )
    return tsne.fit_transform(embeddings)


def draw_interactive_scatter_plot(
    texts: np.ndarray, xs: np.ndarray, ys: np.ndarray, values: np.ndarray
) -> Any:
    # Normalize values to range between 0-255, to assign a color for each value
    max_value = values.max()
    min_value = values.min()
    values_color = (
        ((values - min_value) / (max_value - min_value) * 255)
        .round()
        .astype(int)
        .astype(str)
    )
    values_color_set = sorted(values_color)

    values_list = values.astype(str).tolist()
    values_set = sorted(values_list)

    source = ColumnDataSource(data=dict(x=xs, y=ys, text=texts, perplexity=values_list))
    hover = HoverTool(
        tooltips=[("Sentence", "@text{safe}"), ("Perplexity", "@perplexity")]
    )
    p = figure(plot_width=1200, plot_height=1200, tools=[hover], title="Sentences")
    p.circle(
        "x",
        "y",
        size=10,
        source=source,
        fill_color=factor_cmap(
            "perplexity",
            palette=[Pallete[int(id_)] for id_ in values_color_set],
            factors=values_set,
        ),
    )
    return p


def generate_plot(tsv: str, output_file_name: str, sample: Optional[int]):
    logger.info("Loading dataset in memory")
    df = pd.read_csv(tsv, sep="\t")
    if sample:
        df = df.sample(sample, random_state=SEED)
    logger.info(f"Dataset contains {df.shape[0]} sentences")
    embeddings = df[
        sorted(
            (col for col in df.columns if col.startswith("dim")),
            key=lambda x: int(x.split("_")[-1]),
        )
    ].values
    logger.info(f"Running t-SNE")
    tsne_embeddings = get_tsne_embeddings(embeddings)
    logger.info(f"Generating figure")
    plot = draw_interactive_scatter_plot(
        df["sentence"].values,
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        df["perplexity"].values,
    )
    output_file(output_file_name)
    save(plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embeddings t-SNE plot")
    parser.add_argument(
        "--tsv",
        type=str,
        help="Path to tsv file with columns 'text', 'perplexity' and N 'dim_<i> columns for each embdeding dimension.'",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output HTML file for the interactive plot.",
        default="perplexity_colored_embeddings.html",
    )
    parser.add_argument(
        "--sample", type=int, help="Number of sentences to use", default=None
    )

    args = parser.parse_args()
    generate_plot(args.tsv, args.output_file, args.sample)

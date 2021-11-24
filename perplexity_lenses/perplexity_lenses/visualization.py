import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def draw_histogram(
    values: np.ndarray,
    cutoff_x_axis: float = 2000.0,
    title: str = "Perplexity histogram",
    xlabel: str = "Perplexity",
) -> matplotlib.figure.Figure:
    hist_values = values[values < cutoff_x_axis]
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.hist(hist_values, bins=50)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    return fig

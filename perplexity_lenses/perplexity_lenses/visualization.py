import numpy as np
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Cividis256 as Pallete
from bokeh.plotting import Figure, figure
from bokeh.transform import factor_cmap


def draw_interactive_scatter_plot(
    texts: np.ndarray, xs: np.ndarray, ys: np.ndarray, values: np.ndarray, labels: np.ndarray, text_column: str, label_column: str
) -> Figure:
    # Smooth down values for coloring, by taking the entropy = log10(perplexity) and multiply it by 10000
    values = ((np.log10(values)) * 10000).round().astype(int)
    # Normalize values to range between 0-255, to assign a color for each value
    max_value = values.max()
    min_value = values.min()
    if max_value - min_value == 0:
        values_color = np.ones(len(values))
    else:
        values_color = ((values - min_value) / (max_value - min_value) * 255).round().astype(int)
    values_color_sorted = sorted(values_color)

    values_list = values.astype(str).tolist()
    values_sorted = sorted(values_list)
    labels_list = labels.astype(str).tolist()

    source = ColumnDataSource(data=dict(x=xs, y=ys, text=texts, label=values_list, original_label=labels_list))
    hover = HoverTool(tooltips=[(text_column, "@text{safe}"), (label_column, "@original_label")])
    p = figure(plot_width=800, plot_height=800, tools=[hover])
    p.circle("x", "y", size=10, source=source, fill_color=factor_cmap("label", palette=[Pallete[id_] for id_ in values_color_sorted], factors=values_sorted))

    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.toolbar.logo = None
    return p

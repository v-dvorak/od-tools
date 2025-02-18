from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn

from ..Splitting.SplitUtils import colors


def plot_rectangles(
        rectangles: list[tuple[int, tuple[float, float]]],
        output_path: Path | str = None
) -> None:
    """
    Plot rectangles given by a list of `(class_id, (width, height))`, coordinates are relative,
    to a graph. All rectangles are centered at `(0.5, 0.5)`.

    :param rectangles: list of tuples `(class_id, (width, height))`
    :param output_path: output path, if not None, graph will be saved here
    """
    # create plot
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # plot each rectangle
    for class_id, (width, height) in rectangles:
        rect = patches.Rectangle(
            (0.5 - width / 2, 0.5 - height / 2), width, height,
            linewidth=1,
            edgecolor=colors[class_id],
            facecolor='none'
        )
        ax.add_patch(rect)

    # set labels
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("x")
    plt.ylabel("y")

    # show / save
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path, bbox_inches='tight')


def plot_2d_heatmap(
        coordinates: list[tuple[float, float]],
        num_bins: int = 50,
        xlabel: str = 'x',
        ylabel: str = 'y',
        output_path: Path | str = None
) -> None:
    """
    Plots given 2D data to bins given by a list of `(x, y)`,

    :param coordinates: list of tuples `(x, y)`
    :param num_bins: number of bins, default is 50
    :param xlabel: x label
    :param ylabel: y label
    :param output_path: output path, if not None, graph will be saved here
    """
    # separate x and y values
    x_values = [x for (x, y) in coordinates]
    y_values = [y for (x, y) in coordinates]

    # create plot
    plt.figure(figsize=(6, 6))
    seaborn.histplot(x=x_values, y=y_values, bins=num_bins, pmax=0.9, binrange=[[0, 1], [0, 1]])

    # set labels
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # show / save
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path, bbox_inches='tight')

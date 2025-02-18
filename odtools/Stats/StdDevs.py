from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_stddev(
        means: list[float],
        std_devs: list[float],
        title: str = None,
        names: list[str | int] = None,
        output_path: Path | str = None,
) -> None:
    """
    Plot means wíth standard deviations.

    :param means: means of each column
    :param std_devs: standard deviation of each column
    :param title: title of the plot
    :param names: names of each column
    :param output_path: output path, if not None, graph will be saved here
    """
    # plot values
    x = np.arange(len(means))
    plt.figure(figsize=(10, 6))
    plt.bar(x, height=means, yerr=std_devs, ecolor="red", capsize=5, zorder=10)

    # limits on axis
    plt.xlim(-0.6, len(means) - 0.4)
    plt.ylim(bottom=0)

    # legend
    if title is not None:
        plt.title(title)
    if names is not None:
        plt.xticks(x, names, rotation=90, ha="center")
    else:
        plt.xticks(x, range(len(means)))

    plt.grid(True)
    plt.rc("axes", axisbelow=True)
    plt.rcParams['axes.axisbelow'] = True
    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path, bbox_inches='tight')

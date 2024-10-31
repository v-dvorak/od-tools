import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..Conversions.MungToCOCO import process_mung_page_to_coco


def load_and_plot_stats(
        images_path: Path,
        annotations_path: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        image_format: str = "jpg",
        output_path: Path = None,
        verbose: bool = False,
):
    """
    Load dataset annotations, plot means wíth standard deviations, optionally save to file.

    :param images_path: path to directory with images
    :param annotations_path: path to directory with labels

    :param class_reference_table: dictionary, a function that assigns class id by class name
    :param class_output_names: list of class names

    :param image_format: annot_format in which the images are saved
    :param output_path: output path, if not None, graph will be saved here

    :param verbose: make script verbose
    """
    # load data from given paths
    images = sorted(list(images_path.rglob(f"*.{image_format}")))
    annotations = sorted(list(annotations_path.rglob(f"*.xml")))
    data = list(zip(images, annotations))
    counts = [[] for _ in range(len(class_output_names))]

    print("Loading annotations...")
    for path_to_image, path_to_annotations in tqdm(data, disable=not verbose):
        page = process_mung_page_to_coco(
            path_to_image,
            path_to_annotations,
            class_reference_table,
            class_output_names
        )

        for i in range(len(counts)):
            counts[i].append(len(page.annotations[i]))

    means = []
    stdevs = []
    for i, count in enumerate(counts):
        means.append(statistics.mean(count))
        stdevs.append(statistics.stdev(count))
        if verbose:
            print(f"class: {i}, {class_output_names[i]}")
            print(f"mean: {means[-1]}")
            print(f"stdev: {stdevs[-1]}")

    _plot_stddev(
        means,
        stdevs,
        names=class_output_names,
        output_path=output_path
    )


def _plot_stddev(
        means: list[float],
        std_devs: list[float],
        names: list[str | int] = None,
        output_path: Path = None,
) -> None:
    """
    Plot means wíth standard deviations.

    :param means: means of each column
    :param std_devs: standard deviation of each column
    :param names: names of each column
    :param output_path: output path, if not None, graph will be saved here
    """
    # plot values
    x = np.arange(len(means))
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, means, yerr=std_devs, fmt='o', ecolor='red', capsize=5)

    # limits on axis
    plt.xlim(-0.5, len(means) - 0.5)
    plt.ylim(bottom=0)

    # legend
    plt.title("Average number of annotations on single page")
    if names is not None:
        plt.xticks(x, names, rotation=45, ha="right")
        plt.subplots_adjust(bottom=0.3)
    else:
        plt.xticks(x, range(len(means)))

    plt.grid(True)
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path)

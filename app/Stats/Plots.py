import random
import statistics
from pathlib import Path

from prettytable import PrettyTable, MARKDOWN
from tqdm import tqdm

from . import StdDevs, Bins
from .StatJob import StatJob
from ..Conversions.Annotations import FullPage
from ..Conversions.Formats import InputFormat

SMALL_SIZE_CAP = 32 * 32
MEDIUM_SIZE_CAP = 96 * 96


def load_and_plot_stats(
        images_path: Path,
        annotations_path: Path,
        input_format: InputFormat,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        jobs: list[StatJob] = None,
        summarize: bool = False,
        image_format: str = "jpg",
        output_dir: Path = None,
        seed: int = 42,
        verbose: bool = False,
):
    """
    Load dataset annotations, plot means wíth standard deviations, optionally save to file.

    :param images_path: path to directory with images
    :param annotations_path: path to directory with labels
    :param input_format: input format

    :param class_reference_table: dictionary, a function that assigns class id by class name
    :param class_output_names: list of class names

    :param jobs: list of jobs to perform on data
    :param summarize: whether to add "all" category to statistics

    :param image_format: annot_format in which the images are saved
    :param output_dir: output path, if not None, graph will be saved here

    :param seed: seed for reproducibility
    :param verbose: make script verbose
    """
    # set up params
    if jobs is None:
        jobs = StatJob.get_all()
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)

    # load data from given paths
    images = sorted(list(images_path.rglob(f"*.{image_format}")))
    annotations = sorted(list(annotations_path.rglob(f"*.{input_format.to_annotation_extension()}")))
    data = list(zip(images, annotations))

    # setup variables for stats
    counts = [[] for _ in range(len(class_output_names))]
    all_counts = []

    sizes = [[0, 0, 0] for _ in range(len(annotations))]

    wh_relative_coords = []
    xy_center_relative_coords = []
    class_ids_in_order = []

    # retrieve data for every page in data
    annot_index = 0
    for path_to_image, path_to_annotations in tqdm(data, disable=not verbose, desc="Loading annotations"):
        page = FullPage.load_from_file(
            path_to_annotations,
            path_to_image,
            class_reference_table,
            class_output_names,
            input_format
        )

        # STDDEV job
        if StatJob.ANNOTATION_COUNT_ON_PAGE in jobs:
            for i in range(len(counts)):
                counts[i].append(len(page.annotations[i]))

            if summarize:
                all_counts.append(page.annotation_count())

        # WHBIN, RECT and XYBIN job
        for annot in page.all_annotations():
            if StatJob.WH_HEATMAP in jobs or StatJob.RECTANGLE_PLOT in jobs:
                wh_relative_coords.append(
                    (annot.bbox.width / page.size[0],
                     annot.bbox.height / page.size[1])
                )
            if StatJob.XY_HEATMAP in jobs:
                xy_center_relative_coords.append(
                    ((annot.bbox.left - annot.bbox.width / 2) / page.size[0],
                     (annot.bbox.top - annot.bbox.height / 2) / page.size[1])
                )
            if StatJob.RECTANGLE_PLOT in jobs:
                class_ids_in_order.append(annot.class_id)
            if StatJob.ANNOTATION_SIZES_ON_PAGE in jobs:
                box_area = annot.bbox.area()
                if box_area < SMALL_SIZE_CAP:
                    sizes[annot_index][0] += 1
                elif box_area < MEDIUM_SIZE_CAP:
                    sizes[annot_index][1] += 1
                else:
                    sizes[annot_index][2] += 1

        annot_index += 1

    for job in jobs:
        output_path = None
        if output_dir is not None:
            output_path = output_dir / f"{job.value}.png"

        match job:
            case StatJob.ANNOTATION_COUNT_ON_PAGE:
                _process_stddev(
                    counts,
                    all_counts,
                    class_output_names,
                    title="Average number of annotations per page",
                    summarize=summarize,
                    verbose=verbose,
                    output_path=output_path,
                )
            case StatJob.ANNOTATION_SIZES_ON_PAGE:
                _process_stddev(
                    [[sizes[i][j] for i in range(len(sizes))] for j in range(len(sizes[0]))],
                    None,
                    ["small", "medium", "large"],
                    title="Average annotation sizes per page",
                    output_path=output_path,
                    verbose=verbose
                )
            case StatJob.XY_HEATMAP:
                Bins.plot_2d_heatmap(
                    xy_center_relative_coords,
                    num_bins=50,
                    xlabel="x",
                    ylabel="y",
                    output_path=output_path,
                )
            case StatJob.WH_HEATMAP:
                Bins.plot_2d_heatmap(
                    wh_relative_coords,
                    num_bins=50,
                    xlabel="width",
                    ylabel="height",
                    output_path=output_path,
                )
            case StatJob.RECTANGLE_PLOT:
                data = list(zip(class_ids_in_order, wh_relative_coords))
                random.Random(seed).shuffle(data)
                Bins.plot_rectangles(
                    data[:500],
                    output_path=output_path,
                )


def _process_stddev(
        counts: list[list[int]],
        all_counts: list[int] | None,
        class_output_names: list[str],
        title: str = None,
        summarize: bool = False,
        output_path: Path | str = None,
        verbose: bool = False,
):
    means = []
    stdevs = []

    if verbose:
        table = PrettyTable(["ID", "Name", "Mean", "Stddev"])
        table.set_style(MARKDOWN)
        table.align = "r"

    for i, count in enumerate(counts):
        means.append(statistics.mean(count))
        stdevs.append(statistics.stdev(count))
        if verbose:
            table.add_row([i, class_output_names[i], f"{means[-1]:.4f}", f"{stdevs[-1]:.4f}"])

    if summarize:
        means.append(statistics.mean(all_counts))
        stdevs.append(statistics.stdev(all_counts))
        if verbose:
            table.add_row([-1, "ALL", f"{means[-1]:.4f}", f"{stdevs[-1]:.4f}"])
            table.sortby = "ID"

    if verbose:
        print(title)
        print(table)
        print()

    StdDevs.plot_stddev(
        means,
        stdevs,
        title=title,
        names=class_output_names + ["ALL"] if summarize else class_output_names,
        output_path=output_path
    )

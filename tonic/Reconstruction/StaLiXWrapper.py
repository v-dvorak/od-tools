from pathlib import Path

import cv2
import numpy as np
from stalix import compute_shift_for_measure

from odtools.Conversions.BoundingBox import BoundingBox
from .Graph import Node


def refactor_measures_on_page(
        measures: list[Node],
        bw_image: np.ndarray | str | Path,
        bin_threshold: int = 200,
        space_stddev_threshold: float = 0.02,
        shift_threshold_factor: float = 0.25,
        verbose: bool = False,
        visualize: bool = False
):
    """
    Goes over all given measures and refactors them according to detected staff lines.

    :param measures: measures to be refactored
    :param bw_image: loaded gray image or path to image
    :param bin_threshold: threshold to use for binarization
    :param space_stddev_threshold: found staff lines with stddev of their spaces above this threshold will be ignored
    :param shift_threshold_factor: shifts larger than this fraction of the measure height will be ignored
    :param verbose: make script verbose
    :param visualize: visualize process
    """
    # skip loading image when no measures were found
    if len(measures) == 0:
        return

    if isinstance(bw_image, str) or isinstance(bw_image, Path):
        loaded_image: np.ndarray = cv2.imread(str(bw_image), cv2.IMREAD_GRAYSCALE)
    else:
        loaded_image: np.ndarray = bw_image

    for measure in measures:
        bbox = measure.annot.bbox
        cropped_image = loaded_image[bbox.top:bbox.bottom, bbox.left:bbox.right]
        top_shift, bottom_shift = compute_shift_for_measure(
            cropped_image,
            bin_threshold=bin_threshold,
            space_stddev_threshold=space_stddev_threshold,
            shift_threshold_factor=shift_threshold_factor,
            verbose=verbose,
            visualize=visualize
        )
        measure.annot.bbox = BoundingBox(bbox.left, bbox.top + top_shift, bbox.width, bbox.height - bottom_shift)

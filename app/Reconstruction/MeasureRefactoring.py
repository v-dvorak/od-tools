from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

from ..Conversions.Annotations.Annotation import Annotation
from ..Conversions.BoundingBox import BoundingBox
from ..Reconstruction.Graph import Node


def binarize_image_from_array(img: np.ndarray, threshold=200) -> np.ndarray:
    """
    Convert the image to binary using a given threshold.

    :param img: image to be binarized, already loaded
    :param threshold: threshold to use for binarization
    :return: binarized image
    """
    binary = np.where(img < threshold, 255, 0).astype(np.uint8)
    return binary


def find_peaks_binary(binary_img: np.ndarray) -> np.ndarray:
    """
    Find peaks in rows of binary image.
    """
    black_pixel_counts = np.sum(binary_img == 255, axis=1)
    # find peaks (suspicious regions) using scipy
    valleys, properties = find_peaks(black_pixel_counts, prominence=100)

    # search fails
    if valleys is None or np.size(valleys) < 5:
        return np.array([])

    # np.size(valleys) >= 5
    else:
        valleys: np.ndarray
        top5_indices = np.argpartition(properties["prominences"], -5)[-5:]
        deepest_valleys = valleys[top5_indices]

    return np.sort(deepest_valleys)


def staff_space_stddev(staff_y: np.ndarray) -> np.floating:
    """
    Normalizes given staff line y coordinates and computes their standard deviation.

    :param staff_y: staff line y coordinates
    :return: normalized std dev
    """
    assert len(staff_y) > 1 and staff_y.ndim == 1
    normalized = (staff_y - staff_y[0]) / staff_y[-1]
    diff = np.diff(normalized)
    return np.std(diff)


def check_proposed_staff_lines(
        staff_y: np.ndarray,
        crop_height: int,
        space_stddev_threshold: float = 0.02,
        shift_threshold_factor: float = 0.25,
        verbose: bool = False
) -> bool:
    """
    Checks proposed y coordinates for staff lines. Returns true if
    there are five staff lines,
    std dev of spaces between them is less than the threshold
    and if the proposed shifts of bounding box are less than the threshold.

    :param staff_y: staff line y coordinates
    :param crop_height: height of the cropped images (bbox height)
    :param space_stddev_threshold: threshold of std dev of staff line spacing
    :param shift_threshold_factor: shift threshold factor
    :param verbose: make script verbose
    :return: true if proposed staff lines pass all checks
    """
    # valid staff line system
    if len(staff_y) == 5 and staff_space_stddev(staff_y) < space_stddev_threshold:
        top_offset = staff_y[0]
        bottom_offset = staff_y[-1]
        top_change, bottom_change = top_offset, crop_height - bottom_offset
        shift_limit = crop_height * shift_threshold_factor
        # shift is too large, discard changes
        if top_change > shift_limit or bottom_change > shift_limit:
            if verbose:
                print(f"Proposed shift is too large: {top_change} > {shift_limit} or {bottom_change} > {shift_limit}")
            return False
        else:
            return True
    else:
        if verbose:
            print(
                f"Missing staff lines or std dev over threshold:"
                f"{len(staff_y)}, {staff_space_stddev(staff_y) if len(staff_y) > 1 else 'std err'}"
            )
        return False


def refactor_single_measure(
        measure_annot: Annotation,
        cropped_image: np.ndarray,
        bin_threshold: int = 200,
        space_stddev_threshold: float = 0.02,
        shift_threshold_factor: float = 0.25,
        verbose: bool = False
):
    """
    Refactors single measure according to detected staff lines.

    :param measure_annot: measure to be refactored
    :param cropped_image: loaded gray image
    :param bin_threshold: threshold to use for binarization
    :param space_stddev_threshold: found staff lines with stddev of their spaces above this threshold will be ignored
    :param shift_threshold_factor: shifts larger than this fraction of the measure height will be ignored
    :param verbose: make script verbose
    """
    bbox = measure_annot.bbox
    crop_height = bbox.height
    binary = binarize_image_from_array(cropped_image, threshold=bin_threshold)
    staff_y = find_peaks_binary(binary)
    if check_proposed_staff_lines(
            staff_y,
            crop_height,
            space_stddev_threshold=space_stddev_threshold,
            shift_threshold_factor=shift_threshold_factor,
            verbose=verbose
    ):
        new_bbox = BoundingBox(bbox.left, bbox.top + staff_y[0], bbox.right, bbox.top + staff_y[-1])
        measure_annot.bbox = new_bbox

        if verbose:
            print(f"Refactoring {bbox} to {new_bbox}")


def refactor_measures_on_page(
        measures: list[Node],
        bw_image: np.ndarray | str | Path,
        bin_threshold: int = 200,
        space_stddev_threshold: float = 0.02,
        shift_threshold_factor: float = 0.25,
        verbose: bool = False
):
    """
    Goes over all given measures and refactors them according to detected staff lines.

    :param measures: measures to be refactored
    :param bw_image: loaded gray image or path to image
    :param bin_threshold: threshold to use for binarization
    :param space_stddev_threshold: found staff lines with stddev of their spaces above this threshold will be ignored
    :param shift_threshold_factor: shifts larger than this fraction of the measure height will be ignored
    :param verbose: make script verbose
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
        refactor_single_measure(
            measure.annot,
            cropped_image,
            bin_threshold=bin_threshold,
            space_stddev_threshold=space_stddev_threshold,
            shift_threshold_factor=shift_threshold_factor,
            verbose=verbose
        )

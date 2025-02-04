import math
import warnings
from pathlib import Path

import cv2
import numpy as np

from ..Conversions.BoundingBox import BoundingBox


def create_split_images(image: str | Path | np.ndarray, splits: list[list[BoundingBox]]):
    if isinstance(image, str) or isinstance(image, Path):
        image = cv2.imread(str(image))

    output = []
    for split in [x for xs in splits for x in xs]:
        output.append(image[split.top:split.bottom, split.left:split.right])
    return output


def split_section_to_starts(
        total_length: int,
        section_length: int,
        min_overlap: int,
        adjust_last: bool = False,
) -> list[int]:
    """
    Generates a list of start positions so that section of given length cover the whole length
    with constant overlap that is greater or equal to min_overlap.

    :param total_length: total length to cover
    :param section_length: length of section
    :param min_overlap: minimum overlap between sections
    :param adjust_last: adjusts last box to span exactly to the end (rounding to ints may cause shifts)
    :return: list of start positions
    """
    if section_length > total_length:
        warnings.warn("Section length is bigger than the total length. Returning [0].")
        return [0]

    # maximum number of sections based on the minimal overlap
    number_of_sections = math.ceil((total_length - min_overlap) / (section_length - min_overlap))
    # calculate the step
    step = (total_length - section_length) / (number_of_sections - 1)
    # generate positions
    output = [int(i * step) for i in range(number_of_sections)]

    if adjust_last:
        output[-1] = total_length - section_length

    return output


def create_split_box_matrix(
        image_size: tuple[int, int],
        window_size: tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.25,
) -> list[list[BoundingBox]]:
    """
    Based on the `split_section_to_starts` creates BoundingBox subpages that have `window_size` dimensions
    and neighbouring annotations overlap by minimum of `overlap_ratio`.

    :param image_size: (width, height) of path_to_image
    :param window_size: (width, height) of sliding window, output annotations
    :param overlap_ratio:
    :return: BoundingBox subpages
    """
    img_width, img_height = image_size
    win_width, win_height = window_size

    left_starts = split_section_to_starts(img_width, win_width, int(win_width * overlap_ratio), adjust_last=True)
    top_starts = split_section_to_starts(img_height, win_height, int(win_height * overlap_ratio), adjust_last=True)

    boxes_matrix = []
    for top in top_starts:
        row = []
        for left in left_starts:
            row.append(BoundingBox(left, top, left + win_width, top + win_height))
        boxes_matrix.append(row)

    return boxes_matrix

import math
import warnings
from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from ..Conversions.BoundingBox import BoundingBox


def split_section_to_starts(
    total_length: int,
    section_length: int,
    min_overlap: int,
    adjust_last: bool = False,
) -> list[int]:
    if section_length <= 0:
        raise ValueError(f"section_length must be > 0, got {section_length}")
    if min_overlap < 0:
        raise ValueError(f"min_overlap must be >= 0, got {min_overlap}")
    if min_overlap >= section_length:
        raise ValueError(
            f"min_overlap ({min_overlap}) must be < section_length ({section_length})"
        )

    if section_length >= total_length:
        warnings.warn(
            f"section_length ({section_length}) >= total_length ({total_length}). "
            "Returning [0]."
        )
        return [0]

    number_of_sections = math.ceil(
        (total_length - min_overlap) / (section_length - min_overlap)
    )

    if number_of_sections < 2:
        return [0]

    step = (total_length - section_length) / (number_of_sections - 1)
    positions = [round(i * step) for i in range(number_of_sections)]

    if adjust_last:
        last = total_length - section_length
        # If rounding pushed second-to-last too close, drop the redundant entry
        if len(positions) >= 2 and last <= positions[-2]:
            positions = positions[:-1]
        else:
            positions[-1] = last

    return positions


def create_split_box_matrix(
    image_size: tuple[int, int],
    window_size: tuple[int, int] = (640, 640),
    overlap_ratio: float = 0.25,
) -> list[list[BoundingBox]]:
    """
    Builds a 2-D grid of BoundingBoxes that tile the image with overlapping
    windows.

    :param image_size: `(width, height)` of the source image in pixels.
    :param window_size: `(width, height)` of each output window.
    :param overlap_ratio: Fraction of the window size used as minimum overlap.
        Clamped to `[0.0, 0.99]`.

    :return: Row-major `list[list[BoundingBox]]` where `result[row][col]` is the box
        for that grid position.  Boxes may extend up to `window_size` beyond
        the image boundary on the far edges.
    """
    overlap_ratio = max(0.0, min(0.99, overlap_ratio))

    img_width, img_height = image_size
    win_width, win_height = window_size

    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}")
    if win_width <= 0 or win_height <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    left_starts = split_section_to_starts(
        img_width, win_width, int(win_width * overlap_ratio), adjust_last=True
    )
    top_starts = split_section_to_starts(
        img_height, win_height, int(win_height * overlap_ratio), adjust_last=True
    )

    return [
        [
            BoundingBox(left, top, left + win_width, top + win_height)
            for left in left_starts
        ]
        for top in top_starts
    ]


def get_window_count(
    image_size: tuple[int, int],
    window_size: tuple[int, int] = (640, 640),
    overlap_ratio: float = 0.25,
) -> tuple[int, int]:
    """
    Returns `(cols, rows)` - the number of windows along each axis - without
    building the full box matrix.  Useful for pre-flight memory estimates.
    """
    matrix = create_split_box_matrix(image_size, window_size, overlap_ratio)
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    return cols, rows


def create_split_images(
    image: str | Path | np.ndarray,
    splits: list[list[BoundingBox]],
    include_indices: bool = False,
) -> list[np.ndarray] | list[tuple[int, int, np.ndarray]]:
    """
    Crops an image into windows defined by a BoundingBox matrix.

    :param image: Path to an image file or a pre-loaded numpy array
        (H x W x C, BGR).
    :param splits: Row-major grid of BoundingBoxes as returned by
        `create_split_box_matrix`.
    :param include_indices: If True, returns (row, col, crop) tuples instead of
        bare arrays - handy for reassembling results later.

    :return: Flat list of cropped numpy arrays, or (row, col, array) tuples when
        `include_indices=True`.
    """
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        loaded = cv2.imread(str(path))
        if loaded is None:
            raise ValueError(f"cv2.imread could not decode: {path}")
        image = loaded

    if not splits or not splits[0]:
        raise ValueError("`splits` must be a non-empty 2-D list of BoundingBoxes.")

    img_h, img_w = image.shape[:2]
    output = []

    for row_idx, row in enumerate(splits):
        for col_idx, box in enumerate(row):
            # Clamp to actual image bounds so callers never get empty arrays
            top = max(0, box.top)
            bottom = min(img_h, box.bottom)
            left = max(0, box.left)
            right = min(img_w, box.right)

            crop = image[top:bottom, left:right]

            if include_indices:
                output.append((row_idx, col_idx, crop))
            else:
                output.append(crop)

    return output

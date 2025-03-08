from odtools import BoundingBox, Direction
from .utils import coincides_with_any


def setup_groupings(
        groupings: list[BoundingBox],
        iou_threshold: float = 0.6,
        verbose: bool = False
) -> list[BoundingBox]:
    """
    Removes overlapping groupings based on IoU.
    Prefers bounding boxes that are smaller in area.

    :param groupings: list of bounding boxes
    :param iou_threshold: overlap threshold
    :param verbose: make script verbose
    """
    # smaller bboxes are preferred
    groupings = sorted(groupings, key=lambda g: g.area())

    cleared = []
    for grouping in groupings:
        if coincides_with_any(grouping, cleared, iou_threshold, direction=Direction.VERTICAL):
            if verbose:
                print("Dropping grouping")
            continue
        else:
            cleared.append(grouping)
    return cleared

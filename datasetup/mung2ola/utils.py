from typing import Any, TypeVar

from mung.graph import Node

from odtools import Direction, BoundingBox, Annotation
from .VerticalLine import VerticalLine

STAFF_LINE_COUNT = 5

BB1 = TypeVar("BB1", BoundingBox, VerticalLine)


def _bbox_from_mung_node(node: Node) -> BoundingBox:
    return BoundingBox(node.left, node.top, node.right, node.bottom)


def bboxes_to_annots(data: list[list[BoundingBox]]) -> list[list[Annotation]]:
    """
    Converts a list of bounding boxes to a list of annotations.
    Class IDs of outputted annotations are based on their position in the given list of lists.

    :param data: list of lists of bounding boxes of the same class
    :return: list of lists of annotations of the same class
    """
    all_annots: list[list[Annotation]] = []

    for class_id, boxes in enumerate(data):
        class_annots = []
        for box in boxes:
            class_annots.append(
                Annotation.from_bbox(
                    class_id,
                    box
                )
            )
        all_annots.append(class_annots)

    return all_annots


def page_to_bboxes(page: list[Node], wanted_ids: list[list[str]]) -> list[list[BoundingBox]]:
    def _filter_out_single_id(page: list[Node], ids: list[str]) -> list[BoundingBox]:
        output = []
        for node in page:
            if node.class_name in ids:
                output.append(_bbox_from_mung_node(node))

        return output

    output: list[list[BoundingBox]] = []
    for ids in wanted_ids:
        ids: list[str]
        if isinstance(ids, str):
            ids = [ids]
        output.append(_filter_out_single_id(page, ids))

    return output


class _TempEmptyAnnotation:
    def __init__(self, measure: BoundingBox):
        self.measure = measure
        self._is_empty: bool = True

    def set_not_empty(self):
        self._is_empty = False

    def is_empty(self) -> bool:
        return self._is_empty


def sort_to_strips_with_threshold(
        annots: list[BB1],
        iou_threshold: float,
        direction: Direction = Direction.HORIZONTAL
) -> list[list[BB1]]:
    """
    Sort objects according to the given threshold into strips (rows/columns).
    HORIZONTAL corresponds to the reading order: left to right, top to bottom.
    VERTICAL corresponds to the reading order: bottom to top, left to right.

    :param annots: list of objects to sort
    :param iou_threshold: threshold for sorting, how big there could be between two objects in the same strip
    :param direction: the direction of sorting
    :return: list of sorted strips
    """
    if len(annots) == 0:
        return []

    if direction == Direction.HORIZONTAL:
        top_sorted = sorted(annots, key=lambda a: a.top)
        sorted_rows: list[list[BoundingBox]] = []
        row: list[BoundingBox] = []
        for annot in top_sorted:
            if len(row) == 0:
                row.append(annot)
                continue

            computed_iou = annot.intersection_over_union(
                row[-1],
                direction=Direction.VERTICAL
            )

            if computed_iou > iou_threshold:
                row.append(annot)
            else:
                sorted_rows.append(sorted(row, key=lambda a: a.left))
                row = [annot]

        sorted_rows.append(sorted(row, key=lambda a: a.left))

        return sorted_rows

    elif direction == Direction.VERTICAL:
        left_sorted = sorted(annots, key=lambda a: a.left)
        sorted_rows: list[list[BoundingBox]] = []
        row: list[BoundingBox] = []
        for annot in left_sorted:
            if len(row) == 0:
                row.append(annot)
                continue

            computed_iou = annot.intersection_over_union(
                row[-1],
                direction=Direction.HORIZONTAL
            )

            if computed_iou > iou_threshold:
                row.append(annot)
            else:
                sorted_rows.append(sorted(row, key=lambda a: a.top, reverse=True))
                row = [annot]

        sorted_rows.append(sorted(row, key=lambda a: a.top, reverse=True))

        return sorted_rows

    else:
        raise NotImplementedError(f"Not implemented for direction {direction}")


def chunk(input_list: list[Any], chunk_size: int = STAFF_LINE_COUNT) -> list[list[Any]]:
    """
    Splits given list into chunks of given size.
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def is_close_to_any(bbox: BoundingBox, others: list[BoundingBox], close_limit_px: int) -> bool:
    """
    Returns true if the given bounding box is close to any of the given bounding boxes.
    For to boxes to be considered close, they have to vertically intersect
    and their horizontal distance has to be smaller than close_limit_px.

    :param bbox: main bounding box
    :param others: other bounding boxes
    :param close_limit_px: definition of "close" in pixels
    :return: True if bounding box is close to any of the given bounding boxes, False otherwise
    """
    for other in others:
        if (other.intersects(bbox, Direction.VERTICAL)
                and abs(bbox.left - other.left) < close_limit_px):
            return True
    return False


def intersects_with_any(bbox: BoundingBox, others: list[BoundingBox]) -> bool:
    """
    Returns true if the given bounding box intersects with any of the given bounding boxes.

    :param bbox: main bounding box
    :param others: other bounding boxes
    :return: True if bounding box intersects with any of the given bounding boxes, False otherwise
    """
    for staff in others:
        if bbox.intersects(staff):
            return True
    return False


def get_all_intersecting(
        bbox: BoundingBox,
        others: list[BB1],
        direction: Direction = None,
) -> list[BB1]:
    """
    Returns all bounding boxes intersecting with the given bounding box.

    :param bbox: main bounding box
    :param others: other bounding boxes
    :param direction: the direction of intersecting
    :return: all bounding boxes intersecting with the given bounding box
    """
    return [other for other in others if bbox.intersects(other, direction=direction)]


def get_first_intersecting(
        bbox: BoundingBox,
        others: list[BB1],
        direction: Direction = None
) -> BB1 | None:
    """
    Returns first bounding box intersecting with the given bounding box.

    :param bbox: main bounding box
    :param others: other bounding boxes
    :param direction: the direction of intersecting
    :return: first bounding box intersecting with the given bounding box
    """
    for other in others:
        if bbox.intersects(other, direction=direction):
            return other
    return None


def coincides_with_any(
        bbox: BoundingBox,
        others: list[BoundingBox],
        iou_threshold: float,
        direction: Direction = None
) -> bool:
    """
    Returns true if the given bounding box coincides with any of the given bounding boxes.
    Coincides meaning that the IoU of two boxes is over given threshold.

    :param bbox: main bounding box
    :param others: other bounding boxes
    :param iou_threshold: threshold for IoU
    :param direction: the direction of IoU computation
    :return: True if bounding box coincides with any of the given bounding boxes, False otherwise
    """
    for other in others:
        if bbox.intersection_over_union(other, direction=direction) > iou_threshold:
            return True
    return False

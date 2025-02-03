from enum import Enum

from .Node import Node, VirtualNode
from ..Conversions.BoundingBox import Direction


class SectionType(Enum):
    IN_GS = 0
    OUT_GS = 1


def link_measures_inside_grand_staff(
        top_row: list[Node],
        bottom_row: list[Node],
        linkage_iou_threshold: float = 0.5
) -> list[VirtualNode]:
    """
    Takes measures in top and bottom staff of a single grand staff
    and returns a list of linked pairs connected by VirtualNode.
    If a measure does not link with any other measure, it is returned as a single child of VirtualNode.

    Linkage is made if the computed IoU (intersection over union of pairs horizontal coordinates)
    is less than linkage_iou_threshold.

    :param top_row: list of nodes representing the top staff
    :param bottom_row: list of nodes representing the bottom staff
    :param linkage_iou_threshold: threshold for linkage to be made
    """
    top_index = 0
    bottom_index = 0
    print(len(top_row), len(bottom_row))

    linked_measures: list[VirtualNode] = []

    # going from left to right
    while top_index < len(top_row) and bottom_index < len(bottom_row):
        top_measure = top_row[top_index]
        bottom_measure = bottom_row[bottom_index]

        iou = top_measure.annot.bbox.intersection_over_union(bottom_measure.annot.bbox, direction=Direction.HORIZONTAL)
        print(f"iou: {iou}")

        # linkage found
        if iou > linkage_iou_threshold:
            linked_measures.append(VirtualNode([top_measure, bottom_measure]))
            top_index += 1
            bottom_index += 1

        # throw out one measure and advance in its row
        else:
            # drop the leftmost measure
            if top_measure.annot.bbox.left < bottom_measure.annot.bbox.left:
                linked_measures.append(VirtualNode([top_measure]))
                top_index += 1
            else:
                linked_measures.append(VirtualNode([bottom_measure]))
                bottom_index += 1

    if top_index == len(top_row) and bottom_index == len(bottom_row):
        return linked_measures

    # dump the rest of bottom row
    elif top_index == len(top_row):
        while bottom_index < len(bottom_row):
            linked_measures.append(VirtualNode([bottom_row[bottom_index]]))
            bottom_index += 1
    # dump the rest of top row
    else:
        while top_index < len(top_row):
            linked_measures.append(VirtualNode([top_row[top_index]]))
            top_index += 1

    return linked_measures

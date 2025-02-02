from enum import Enum

from .Node import Node, VirtualNode
from ..Conversions.BoundingBox import Direction


class SectionType(Enum):
    IN_GS = 0
    OUT_GS = 1


def sort_page_into_sections(
        measures: list[Node],
        grand_staff: list[Node]
) -> list[tuple[SectionType, list[Node]]]:
    """
    Sorts given list of Measures into two types of sections: in and out of grand stave,
    based on the list of given grand staff.

    :param measures: list of measures to sort
    :param grand_staff: list of measures to sort
    :return: list of tuples (in/out section type, list of nodes)
    """
    grand_staff = sorted(grand_staff, key=lambda g: g.annot.bbox.top)
    measures = sorted(measures, key=lambda m: m.annot.bbox.top)
    sections: list[tuple[SectionType, list[Node]]] = []

    section = []
    gs_index = 0
    in_gs = False

    for measure in measures:
        # ran out of gs to assign measures to, dump them to last section of page
        if gs_index >= len(grand_staff):
            section.append(measure)
            in_gs = False
            continue

        current_gs = grand_staff[gs_index]
        intersects = measure.annot.bbox.intersects(current_gs.annot.bbox)

        # measure is inside gs and it is the first one found
        if intersects and not in_gs:
            # edge case, when the algorithm starts inside gs, this could otherwise create an empty section
            if len(section) > 0:
                sections.append((SectionType.OUT_GS, section))
            section = [measure]
            in_gs = True
        # measure is inside gs and the gs is the same as the last one
        elif intersects and in_gs:
            section.append(measure)
        # measure is outside any gs, same as the last one
        elif not intersects and not in_gs:
            section.append(measure)
        # measure is outside gs and the last one was inside
        elif not intersects and in_gs:
            sections.append((SectionType.IN_GS, section))
            section = [measure]

            # it can intersect with the next gs
            if (gs_index + 1 < len(grand_staff)
                    and measure.annot.bbox.intersects(grand_staff[gs_index + 1].annot.bbox)):
                in_gs = True
            else:
                in_gs = False

            # switch to next gs
            gs_index += 1
        else:
            raise NotImplementedError()

    # append last section
    sections.append((SectionType.IN_GS if in_gs else SectionType.OUT_GS, section))
    return sections


def link_measures_inside_grand_stave(
        top_row: list[Node],
        bottom_row: list[Node],
        linkage_iou_threshold: float = 0.5
) -> list[VirtualNode]:
    """
    Takes measures in top and bottom stave of a single grand stave
    and returns a list of linked pairs connected by VirtualNode.
    If a measure does not link with any other measure, it is returned as a single child of VirtualNode.

    Linkage is made if the computed IoU (intersection over union of pairs horizontal coordinates)
    is less than linkage_iou_threshold.

    :param top_row: list of nodes representing the top stave
    :param bottom_row: list of nodes representing the bottom stave
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

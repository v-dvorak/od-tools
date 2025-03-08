from typing import Any

from odtools import BoundingBox, Direction
from .VerticalLine import VerticalLine
from .utils import sort_to_strips_with_threshold


def staffs_to_systems(
        staffs: list[BoundingBox],
        system_spans: list[VerticalLine]
) -> list[BoundingBox]:
    """
    Finds overlapping staffs for every system span and converts them into a system bounding box.

    :param staffs: list of staffs
    :param system_spans: list of system spans
    :return: list of systems
    """
    if len(staffs) == 0 or len(system_spans) == 0:
        return []

    staffs = sorted(staffs, key=lambda staff: staff.top)
    system_spans = sorted(system_spans, key=lambda middle_mss: middle_mss.top)

    staff_index = 0
    ms_index = 0

    output: list[BoundingBox] = []
    system: list[BoundingBox] = []
    while staff_index < len(staffs) and ms_index < len(system_spans):
        staff = staffs[staff_index]
        ms = system_spans[ms_index]
        if ms.intersects(staff, direction=Direction.VERTICAL):
            system.append(staff)
            staff_index += 1
        else:
            # check if there are any staffs
            if len(system) > 0:
                output.append(BoundingBox.from_list_of_boxes(system))
                system = []
                ms_index += 1
            # staff does not belong to current system, but belongs to the next one
            elif (ms_index + 1 < len(system_spans)
                  and system_spans[ms_index + 1].intersects(staff, direction=Direction.VERTICAL)):
                system = [staff]
                ms_index += 1
                staff_index += 1
            # throw out current staff (it does not belong to current system or the next one)
            else:
                staff_index += 1

    if len(system) > 0:
        output.append(BoundingBox.from_list_of_boxes(system))
    return output


def filter_out_system_spans(
        measure_separators: list[VerticalLine],
        row_threshold: float = 0.3
) -> list[VerticalLine]:
    """
    Sorts separators by reading order and returns the middle one from every row,
    effectively filtering out a vertical span for every system.

    :param measure_separators: list of measure separators
    :param row_threshold: threshold for row sorting
    :return: list of vertical spans of systems on page
    """

    def _get_middle_of_list(lst: list[Any]) -> Any:
        return lst[len(lst) // 2]

    rows = sort_to_strips_with_threshold(
        measure_separators,
        iou_threshold=row_threshold,
        direction=Direction.HORIZONTAL
    )

    # assume the worst (crooked writing) and take the middle measure separator
    # as the representation of the whole system
    output: list[VerticalLine] = []
    for row in rows:
        middle_ms = _get_middle_of_list(row)
        output.append(VerticalLine(0, middle_ms.top, middle_ms.bottom))
    return output

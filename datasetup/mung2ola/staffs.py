from odtools import BoundingBox, Direction
from .utils import STAFF_LINE_COUNT, chunk, _TempEmptyAnnotation


def staff_lines_to_staffs(
        staff_lines: list[BoundingBox],
        staff_line_count: int = STAFF_LINE_COUNT,
) -> list[BoundingBox]:
    """
    Sorts staff lines according to the top of their bounding box,
    splits them into groups of five and converts these group to new bounding boxes.

    Assumes that the number of staff lines is divisible by ``STAFF_LINE_COUNT`` (defaults to 5).

    :param staff_lines: list of staff lines
    :param staff_line_count: number of staff lines inside a single staff
    :return: list of bounding boxes of created staffs
    """
    if len(staff_lines) % staff_line_count != 0:
        print("TODO: fix this, suppressing error")
        staff_lines = sorted(staff_lines, key=lambda a: a.left)[:-1]
        if len(staff_lines) % staff_line_count != 0:
            raise ValueError(f"Staff lines count must be divisible by {staff_line_count},"
                             f"input length is {len(staff_lines)}")

    sl_to_staffs = chunk(sorted(staff_lines, key=lambda s: s.top), chunk_size=staff_line_count)

    staffs = []
    for sls in sl_to_staffs:
        staffs.append(BoundingBox.from_list_of_boxes(sls))

    return staffs


def filter_out_empty_staffs(
        staffs: list[BoundingBox],
        measures: list[BoundingBox],
        iou_threshold: float = 0.8,
        verbose: bool = False
) -> list[BoundingBox]:
    """
    Filter out all staffs that don't intersect with any valid measure.
    """

    def _check_staff_emptiness(staffs: list[_TempEmptyAnnotation], sources: list[BoundingBox]):
        def _check_single_staff_emptiness(staff: _TempEmptyAnnotation, sources: list[BoundingBox]):
            for source in sources:
                if staff.measure.intersection_over_union(source, direction=Direction.VERTICAL) > iou_threshold:
                    staff.set_not_empty()
                    return

        for staff in staffs:
            _check_single_staff_emptiness(staff, sources)

    wrappers = [_TempEmptyAnnotation(s) for s in staffs]
    _check_staff_emptiness(wrappers, measures)

    output = []
    for wrapper in wrappers:
        if wrapper.is_empty():
            if verbose:
                print("No symbols found for staff")
            continue
        else:
            output.append(wrapper.measure)
    return output

from odtools import BoundingBox, Direction
from .VerticalLine import VerticalLine
from .utils import (is_close_to_any, intersects_with_any, get_all_intersecting,
                    get_first_intersecting, sort_to_strips_with_threshold)


def resolve_barlines_from_annots(
        barlines: list[BoundingBox],
        close_limit_px: int = 10,
        verbose: bool = False
) -> list[VerticalLine]:
    """
    :param barlines: list of barlines loaded as Annotations
    :param close_limit_px: barline closer than this to other barlines will be dropped
    :param verbose: make script verbose
    """
    converted = []
    for barline in barlines:
        converted.append(VerticalLine.from_bbox(barline))

    converted.sort(key=lambda b: b.left)

    cleared = []
    for barline in converted:
        if is_close_to_any(barline, cleared, close_limit_px):
            if verbose:
                print("Dropping barline")
            continue
        else:
            cleared.append(barline)

    return cleared


def add_grouping_to_barlines(
        barlines: list[VerticalLine],
        groupings: list[BoundingBox],
        staffs: list[BoundingBox],
        close_limit_px: int = 50,
        verbose: bool = False
) -> list[VerticalLine]:
    for grouping in groupings:
        if is_close_to_any(grouping, barlines, close_limit_px):
            if verbose:
                print("Not adding grouping to the barlines")
            continue

        proposed_barline = VerticalLine.from_bbox(grouping)
        if intersects_with_any(proposed_barline, staffs):
            barlines.append(proposed_barline)

    return barlines


def cut_barlines_to_fit_staffs(
        barlines: list[VerticalLine],
        staffs: list[BoundingBox],
) -> list[VerticalLine]:
    def _cut_barline_to_fit_staff(barline: VerticalLine, staff: BoundingBox) -> VerticalLine:
        return VerticalLine(
            barline.left,
            max(staff.top, barline.top),
            min(staff.bottom, barline.bottom)
        )

    output_barlines = []
    for staff in staffs:
        current_bl = get_all_intersecting(staff, barlines, direction=Direction.VERTICAL)
        for bl in current_bl:
            output_barlines.append(_cut_barline_to_fit_staff(bl, staff))

    return output_barlines


def add_barlines_to_staff_ends(
        rows: list[list[VerticalLine]],
        staffs: list[BoundingBox],
        close_limit_px: int = 50,
        verbose: bool = False
) -> list[list[VerticalLine]]:
    """
    Assumes that every row is sorted in reading order.
    """
    for row in rows:
        # create new leftmost barline
        leftmost_barline = row[0]
        intersecting_staff = get_first_intersecting(leftmost_barline, staffs, direction=Direction.VERTICAL)
        assert intersecting_staff is not None  # barline should always intersect with one staff

        left_barline = VerticalLine(
            intersecting_staff.left,
            leftmost_barline.top,
            leftmost_barline.bottom
        )

        # create new rightmost barline
        rightmost_barline = row[-1]
        right_barline = VerticalLine(
            intersecting_staff.right,
            rightmost_barline.top,
            rightmost_barline.bottom
        )
        for proposed_barline in [left_barline, right_barline]:
            if is_close_to_any(proposed_barline, row, close_limit_px):
                if verbose:
                    print("Not adding staff ends to the barlines")
                continue
            else:
                row.append(proposed_barline)
        row.sort(key=lambda x: x.left)

    return rows


def clean_up_rows_of_barlines(rows: list[list[VerticalLine]], verbose: bool = False) -> list[list[VerticalLine]]:
    output = []
    for row in rows:
        if len(row) < 2:
            row_repre = ", ".join(str(line) for line in row)
            if verbose:
                print(f"Not enough barlines in row {len(row)}: {row_repre}")
        else:
            output.append(row)
    return output


def process_barlines_to_rows(
        barlines: list[VerticalLine],
        staffs: list[BoundingBox],
        iou_threshold: float,
        verbose: bool = False
) -> list[list[VerticalLine]]:
    """
    Sorts barlines to rows, assumes that staff ends are barlines, removes those that might not be valid.

    :param barlines: list of barlines
    :param staffs: list of staffs, their end are potential barlines
    :param iou_threshold: iou threshold
    :param verbose: make script verbose
    :return: list of cleaned up rows of barlines
    """
    rows_of_barlines = sort_to_strips_with_threshold(barlines, iou_threshold, direction=Direction.HORIZONTAL)
    rows_of_barlines = add_barlines_to_staff_ends(rows_of_barlines, staffs, verbose=verbose)
    rows_of_barlines = clean_up_rows_of_barlines(rows_of_barlines, verbose=verbose)
    return rows_of_barlines


def setup_barlines(
        barlines: list[VerticalLine],
        staffs: list[BoundingBox],
        groupings: list[BoundingBox],
        verbose: bool = False
) -> list[VerticalLine]:
    """
    Removes overlapping barlines, assumes that groupings might be barlines,
    cuts all resolved barlines (with groupings added) to fit their overlapping staffs.

    :param barlines: list of barlines to process
    :param staffs: list of staffs to check barline overlaps with
    :param groupings: list of barlines to add to barlines
    :param verbose: make script verbose
    """
    barlines = resolve_barlines_from_annots(barlines, verbose=verbose)
    barlines = add_grouping_to_barlines(barlines, groupings, staffs, verbose=verbose)
    barlines = cut_barlines_to_fit_staffs(barlines, staffs)
    return barlines

from odtools import BoundingBox, Direction


def staffs_to_grand_staffs(
        staffs: list[BoundingBox],
        staff_bars: list[BoundingBox],
        vertical_overlap_threshold: float = 0.5,
        verbose: bool = False
) -> list[BoundingBox]:
    """
    Counts vertically overlapping staffs for every given staff bar and decides if it is a grand staff or not.
    Duplicate grand staffs are removed automatically.

    :param staffs: list of staffs
    :param staff_bars: list of staff bars and groupings
    :param vertical_overlap_threshold: threshold for determining whether a staff belongs to a staff bar
    :param verbose: make script verbose
    :return: list of grand staffs
    """

    def _vertical_overlap(first: BoundingBox, second: BoundingBox) -> int:
        if first.intersects(second, direction=Direction.VERTICAL):
            start = max(first.top, second.top)
            end = min(first.bottom, second.bottom)
            return abs(end - start)
        return 0

    grand_staffs: list[BoundingBox] = []
    for grouping in staff_bars:
        grouped_staffs: list[BoundingBox] = []
        for staff in staffs:
            if _vertical_overlap(staff, grouping) / staff.height > vertical_overlap_threshold:
                grouped_staffs.append(staff)
        if len(grouped_staffs) == 2:
            grand_staffs.append(BoundingBox.from_list_of_boxes(grouped_staffs))

    # filter out duplicates
    start_len = len(grand_staffs)
    grand_staffs = list(set(grand_staffs))

    if verbose:
        print(f"Removed {start_len - len(grand_staffs)} identical groups")

    return grand_staffs

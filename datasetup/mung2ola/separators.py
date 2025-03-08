from odtools import BoundingBox
from .VerticalLine import VerticalLine
from .utils import get_all_intersecting


def refine_measure_separators(
        measure_separators: list[BoundingBox],
        barlines: list[VerticalLine]
) -> list[VerticalLine]:
    """
    Finds all barlines that belong inside the separator bounding box
    and cuts the measure separator to perfectly vertically fit around these barlines.

    :param measure_separators: list of measure separators to refine
    :param barlines: list of barlines to check intersections with
    :return: list of refined barlines
    """

    def _refine_ms(ms: BoundingBox, intersecting_barlines: list[VerticalLine]) -> VerticalLine:
        if len(intersecting_barlines) == 0:  # this should never happen
            return VerticalLine.from_bbox(ms)
        barline_top = min(intersecting_barlines, key=lambda x: x.top).top
        barline_bottom = max(intersecting_barlines, key=lambda x: x.bottom).bottom
        return VerticalLine(
            (ms.left + ms.right) // 2,
            max(barline_top, ms.top),
            min(barline_bottom, ms.bottom)
        )

    refined_mss: list[VerticalLine] = []
    for ms in measure_separators:
        inter = get_all_intersecting(ms, barlines)
        refined_mss.append(_refine_ms(ms, inter))
    return refined_mss

from odtools import BoundingBox
from .VerticalLine import VerticalLine
from .utils import _TempEmptyAnnotation


def row_of_barlines_to_measures(barlines: list[VerticalLine]) -> list[BoundingBox]:
    assert len(barlines) > 1

    measures: list[BoundingBox] = []

    for index in range(len(barlines) - 1):
        bbox = BoundingBox.from_list_of_boxes([barlines[index], barlines[index + 1]])
        measures.append(bbox)

    return measures


def page_of_barlines_to_measures(rows_of_barlines: list[list[VerticalLine]]) -> list[BoundingBox]:
    output: list[BoundingBox] = []
    for row in rows_of_barlines:
        output.extend(row_of_barlines_to_measures(row))

    return output


def filter_out_empty_measures(measures: list[BoundingBox], sources: list[BoundingBox]) -> list[BoundingBox]:
    def _check_measure_emptiness(measures: list[_TempEmptyAnnotation], sources: list[BoundingBox]):
        def _vertical_distance(first: BoundingBox, second: BoundingBox) -> int:
            center1, _ = first.center()
            center2, _ = second.center()
            return int(abs(center1 - center2))

        def _center_intersect_horizontal(measure: BoundingBox, source: BoundingBox) -> bool:
            _, h_center = source.center()
            return measure.left <= h_center <= measure.right

        for source in sources:
            closest_m: _TempEmptyAnnotation = None
            closest_distance: int = float("inf")

            for tm in measures:
                if _center_intersect_horizontal(tm.measure, source):
                    d = _vertical_distance(tm.measure, source)
                    if d < closest_distance:
                        closest_m = tm
                        closest_distance = d

            if closest_m is not None:
                closest_m.set_not_empty()

    wrappers = [_TempEmptyAnnotation(m) for m in measures]
    _check_measure_emptiness(wrappers, sources)

    return [wrapper.measure for wrapper in wrappers if not wrapper.is_empty()]


def filter_out_empty_measures_based_on_system_measures(
        measures: list[BoundingBox],
        system_measures: list[BoundingBox],
        ratio_threshold: float,
        verbose: bool = False
) -> list[BoundingBox]:
    def _inside_ratio(measure: BoundingBox, system_measure: BoundingBox) -> float:
        overlap_area = measure.intersection_area(system_measure)
        ratio = overlap_area / measure.area()
        if verbose:
            if ratio > 0:
                print(f"intersection area / measure area = {ratio}")
        return ratio

    output: list[BoundingBox] = []
    for measure in measures:
        for sm in system_measures:
            if _inside_ratio(measure, sm) > ratio_threshold:
                output.append(measure)

    return output


def rows_of_barlines_to_measures(
        rows_of_barlines: list[list[VerticalLine]],
        system_measures: list[BoundingBox],
        inside_threshold: float
) -> list[BoundingBox]:
    """
    Creates a measure bounding box for every two neighboring barlines.
    Filters out invalid measures,
    measure is considered invalid if it not inside any given valid system measure.

    :param rows_of_barlines: list of rows of separators
    :param system_measures: list of system measures
    :param inside_threshold: how much of a measure has to be inside any system for it co be considered valid
    :return: list of valid measures
    """
    measures = page_of_barlines_to_measures(rows_of_barlines)
    return filter_out_empty_measures_based_on_system_measures(measures, system_measures, inside_threshold)

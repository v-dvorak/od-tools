from odtools import BoundingBox
from .VerticalLine import VerticalLine
from .measures import page_of_barlines_to_measures, filter_out_empty_measures


def rows_of_separators_to_system_measures(
        rows_of_separators: list[list[VerticalLine]],
        other_symbols: list[BoundingBox]
) -> list[BoundingBox]:
    """
    Creates a system measure bounding box for every two neighboring separators.
    Filters out invalid system measures,
    measure is considered invalid if it is not the closest measure for any ``other_symbol``.

    :param rows_of_separators: list of rows of separators
    :param other_symbols: list of symbols that will be assigned to measures
    :return: list of valid systems measures
    """
    system_measures = page_of_barlines_to_measures(rows_of_separators)
    return filter_out_empty_measures(system_measures, other_symbols)

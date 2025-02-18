from ..Utils import ExtendedEnum


class StatJob(ExtendedEnum):
    ANNOTATION_COUNT_ON_PAGE = "counts"
    ANNOTATION_SIZES_ON_PAGE = "sizes"
    XY_HEATMAP = "xybin"
    WH_HEATMAP = "whbin"
    RECTANGLE_PLOT = "rect"

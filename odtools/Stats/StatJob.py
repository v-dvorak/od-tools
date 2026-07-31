from enum import StrEnum


class StatJob(StrEnum):
    ANNOTATION_COUNT_ON_PAGE = "counts"
    ANNOTATION_SIZES_ON_PAGE = "sizes"
    XY_HEATMAP = "xybin"
    WH_HEATMAP = "whbin"
    RECTANGLE_PLOT = "rect"

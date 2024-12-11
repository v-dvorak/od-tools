from enum import Enum
from typing import Self


class StatJob(Enum):
    ANNOTATION_COUNT_ON_PAGE = "stddev"
    ANNOTATION_SIZES_ON_PAGE = "sizes"
    XY_HEATMAP = "xybin"
    WH_HEATMAP = "whbin"
    RECTANGLE_PLOT = "rect"

    @staticmethod
    def get_all_jobs_str() -> list[str]:
        return [
            StatJob.ANNOTATION_COUNT_ON_PAGE.value,
            StatJob.ANNOTATION_SIZES_ON_PAGE.value,
            StatJob.XY_HEATMAP.value,
            StatJob.WH_HEATMAP.value,
            StatJob.RECTANGLE_PLOT.value
        ]

    @staticmethod
    def get_all_jobs() -> list[Self]:
        return [
            StatJob.ANNOTATION_COUNT_ON_PAGE,
            StatJob.ANNOTATION_SIZES_ON_PAGE,
            StatJob.XY_HEATMAP,
            StatJob.WH_HEATMAP,
            StatJob.RECTANGLE_PLOT
        ]

    @classmethod
    def from_string(cls, token: str) -> Self:
        match token.lower():
            case StatJob.ANNOTATION_COUNT_ON_PAGE.value:
                return StatJob.ANNOTATION_COUNT_ON_PAGE
            case StatJob.ANNOTATION_SIZES_ON_PAGE.value:
                return StatJob.ANNOTATION_SIZES_ON_PAGE
            case StatJob.XY_HEATMAP.value:
                return StatJob.XY_HEATMAP
            case StatJob.WH_HEATMAP.value:
                return StatJob.WH_HEATMAP
            case StatJob.RECTANGLE_PLOT.value:
                return StatJob.RECTANGLE_PLOT
            case _:
                raise ValueError(f"Invalid job name: \"{token}\"")


from enum import Enum
from typing import Self


class StatJob(Enum):
    ANNOTATION_COUNT_ON_PAGE = "counts"
    ANNOTATION_SIZES_ON_PAGE = "sizes"
    XY_HEATMAP = "xybin"
    WH_HEATMAP = "whbin"
    RECTANGLE_PLOT = "rect"

    @staticmethod
    def get_all_jobs_value() -> list[str]:
        return [j.value for j in StatJob]

    @staticmethod
    def get_all_jobs() -> list[Self]:
        return [j for j in StatJob]

    @classmethod
    def from_string(cls, token: str) -> Self:
        token = token.lower()
        for j in StatJob:
            if j.value == token:
                return j

        raise ValueError(f"Invalid job name: \"{token}\"")

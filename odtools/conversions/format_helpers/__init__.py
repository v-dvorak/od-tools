from typing import TypeVar

from ..formats import Format
from .base import _HelperBase

# helper modules must be imported BEFORE the registry is built,
# so that their classes can be found
from .coco import COCOHelper
from .dolores_coco import DoloresCOCOHelper
from .mung import MuNGHelper
from .yolod import YOLODetectionHelper
from .yolos import YOLOSegmentationHelper
from .semseg import SemanticSegmentationHelper


T = TypeVar("T")


def all_subclasses(cls: type[T]) -> set[type[T]]:
    """Recursively collects all subclasses of the given class."""
    result: set[type[T]] = set()
    stack: list[type[T]] = list(cls.__subclasses__())

    while stack:
        subclass = stack.pop()
        if subclass not in result:
            result.add(subclass)
            stack.extend(subclass.__subclasses__())

    return result


FORMAT_HELPERS: dict[Format, type[_HelperBase]] = {}
"""Every helper implementing a format, keyed by the format it implements"""

for subclass in all_subclasses(_HelperBase):
    format_type = subclass.FORMAT_TYPE
    if format_type is None:
        # helper does not declare a format yet
        continue
    if format_type in FORMAT_HELPERS:
        raise ValueError(f"Multiple helpers registered for format {format_type}")
    FORMAT_HELPERS[format_type] = subclass

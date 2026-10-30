from __future__ import annotations
from enum import StrEnum


class Format(StrEnum):
    MUNG = "mung"
    COCO = "coco"
    YOLO_DETECTION = "yolod"
    YOLO_SEGMENTATION = "yolos"
    DOLORES_COCO = "dolores"
    SEMANTIC_SEGMENTATION = "semseg"

    def to_annotation_extension(self) -> str:
        match self:
            case Format.MUNG:
                return "xml"
            case Format.COCO | Format.DOLORES_COCO:
                return "json"
            case Format.YOLO_DETECTION | Format.YOLO_SEGMENTATION:
                return "txt"
            case Format.SEMANTIC_SEGMENTATION:
                return "png"
            case _:
                raise ValueError

    @classmethod
    def input_formats(cls) -> list[Format]:
        """Formats that can be loaded."""
        from ..conversions.format_helpers import all_subclasses, _HelperBase

        output: set[Format] = set()
        for sc in all_subclasses(_HelperBase):
            if sc.FORMAT_TYPE is not None and sc.supports_from_file():
                output.add(sc.FORMAT_TYPE)
        return list(output)

    @classmethod
    def output_formats(cls) -> list[Format]:
        """Formats that can be saved."""
        from ..conversions.format_helpers import all_subclasses, _HelperBase

        output: set[Format] = set()
        for sc in all_subclasses(_HelperBase):
            if sc.FORMAT_TYPE is not None and sc.supports_to_file():
                output.add(sc.FORMAT_TYPE)
        return list(output)

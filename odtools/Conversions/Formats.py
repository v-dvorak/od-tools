from enum import StrEnum


class InputFormat(StrEnum):
    MUNG = "mung"
    COCO = "coco"
    YOLO_DETECTION = "yolod"
    YOLO_SEGMENTATION = "yolos"
    DOLORES_COCO = "dolores"

    def to_annotation_extension(self) -> str:
        match self:
            case InputFormat.MUNG:
                return "xml"
            case InputFormat.COCO | InputFormat.DOLORES_COCO:
                return "json"
            case InputFormat.YOLO_DETECTION | InputFormat.YOLO_SEGMENTATION:
                return "txt"
            case _:
                raise ValueError


class OutputFormat(StrEnum):
    MUNG = "mung"
    COCO = "coco"
    YOLO_DETECTION = "yolod"
    YOLO_SEGMENTATION = "yolos"
    SEMANTIC_SEGMENTATION = "semseg"

    def to_annotation_extension(self) -> str:
        match self:
            case OutputFormat.MUNG:
                return "xml"
            case OutputFormat.COCO:
                return "json"
            case OutputFormat.YOLO_DETECTION | OutputFormat.YOLO_SEGMENTATION:
                return "txt"
            case OutputFormat.SEMANTIC_SEGMENTATION:
                return "png"
            case _:
                raise ValueError

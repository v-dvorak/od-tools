from ..Utils import ExtendedEnum


class InputFormat(ExtendedEnum):
    MUNG = "mung"
    COCO = "coco"
    YOLO_DETECTION = "yolod"
    YOLO_SEGMENTATION = "yolos"

    def to_annotation_extension(self) -> str:
        match self:
            case InputFormat.MUNG:
                return "xml"
            case InputFormat.COCO:
                return "json"
            case InputFormat.YOLO_DETECTION | InputFormat.YOLO_SEGMENTATION:
                return "txt"
            case _:
                raise ValueError


class OutputFormat(ExtendedEnum):
    MUNG = "mung"
    COCO = "coco"
    YOLO_DETECTION = "yolod"
    YOLO_SEGMENTATION = "yolos"

    def to_annotation_extension(self) -> str:
        match self:
            case OutputFormat.MUNG:
                return "xml"
            case OutputFormat.COCO:
                return "json"
            case OutputFormat.YOLO_DETECTION | OutputFormat.YOLO_SEGMENTATION:
                return "txt"
            case _:
                raise ValueError

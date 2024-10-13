from pathlib import Path

from mung.node import Node


class YOLODetection:
    def __init__(self, x_center, y_center, width, height, cls: int = None):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.cls = cls

    def __str__(self):
        return f"{self.cls} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class YOLOSegmentation:
    def __init__(self, coordinates: list[tuple[float, float]]):
        self.coordinates = coordinates
        self.cls = None

    def __str__(self):
        coords = " ".join([f"{point[0]:.6f} {point[1]:.6f}" for point in self.coordinates])
        return f"{self.cls} {coords}"


from utils import mung_segmentation_to_absolute_coordinates


class COCOAnnotation:
    def __init__(self, class_id: int, left: int, top: int, width: int, height: int, segmentation: list[tuple[int, int]]):
        self.class_id = class_id
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.segmentation = segmentation

    @classmethod
    def from_mung_node(cls, clss: int, node: Node):
        return cls(clss, node.left, node.top, node.width, node.height, mung_segmentation_to_absolute_coordinates(node))

    def __str__(self):
        return f"({self.class_id=}, {self.left}, {self.top}, {self.width}, {self.height}, {self.segmentation})"

class COCOFullPage:
    def __init__(
            self,
            # source: str | Path,
            image_size: tuple[int, int],
            annotations: list[COCOAnnotation],
            class_names: list[str]
    ):
        """
        :param annotations: list of COCOAnnotation
        :param image_size: image size, (width, height)
        """
        # self.source = source
        self.size = image_size
        self.classes = class_names
        self.annotations = None
        self._sort_annotations_by_class(annotations, len(class_names))

    def _sort_annotations_by_class(self, annotations: list[COCOAnnotation], class_count: int):
        self.annotations = [[] for _ in range(class_count)]
        for annot in annotations:
            self.annotations[annot.class_id].append(annot)

    def __str__(self):
        return f"({self.classes=}, {self.size=}, {self.annotations})"

    # @classmethod
    # def from_mung_page(class_id, annotations: list[COCOAnnotation], image_size: tuple[int, int]):


from json import JSONEncoder


class COCOFullPageEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, COCOFullPage):
            output = {
                # "source": obj.source,
                "width": obj.size[0],
                "height": obj.size[1],
            }
            for i in range(len(obj.classes)):
                output[obj.classes[i]] = obj.annotations[i]
            return output
        elif isinstance(obj, COCOAnnotation):
            return COCOAnnotationEncoder().default(obj)

        return super().default(obj)


class COCOAnnotationEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, COCOAnnotation):
            return {
                "left": obj.left,
                "top": obj.top,
                "width": obj.width,
                "height": obj.height,
                "segmentation": [[", ".join([f"{x}, {y}" for x, y in obj.segmentation])]],
            }
        return super().default(obj)

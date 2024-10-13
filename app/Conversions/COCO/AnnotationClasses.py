from json import JSONEncoder

from mung.node import Node

from .Interfaces import ICOCOAnnotation, ICOCOFullPage
from .. import ConversionUtils


class COCOAnnotation(ICOCOAnnotation):
    def __init__(self, class_id: int, left: int, top: int, width: int, height: int,
                 segmentation: list[tuple[int, int]]):
        super().__init__(class_id, left, top, width, height, segmentation)

    @classmethod
    def from_mung_node(cls, clss: int, node: Node):
        return cls(
            clss,
            node.left, node.top, node.width, node.height,
            ConversionUtils.mung_segmentation_to_absolute_coordinates(node)
        )


class COCOFullPage(ICOCOFullPage):
    def __init__(self, image_size: tuple[int, int], annotations: list[COCOAnnotation], class_names: list[str]):
        super().__init__(image_size, annotations, class_names)


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
            # flatten
            segm = []
            for x, y in obj.segmentation:
                segm.append(x)
                segm.append(y)

            return {
                "left": obj.left,
                "top": obj.top,
                "width": obj.width,
                "height": obj.height,
                "segmentation": [segm],
            }
        return super().default(obj)

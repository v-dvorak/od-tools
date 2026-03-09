import json
from json import JSONEncoder
from pathlib import Path
from typing import TYPE_CHECKING

from ..Annotation import Annotation
from ..annotation_type import AnnotationType
if TYPE_CHECKING:
    from ..FullPage import FullPage


class _COCOHelper:
    @staticmethod
    def from_coco_file(
            file_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> "FullPage":
        from ..FullPage import FullPage
        with open(file_path.__str__(), "r") as file:
            data = json.load(file)
        image_width, image_height = data["width"], data["height"]
        annots = [[] for _ in range(len(class_output_names))]
        for class_name in class_reference_table.keys():
            for annot in data[class_name]:
                # process coordinates
                left = annot["left"]
                top = annot["top"]
                width = annot["width"]
                height = annot["height"]

                # process segmentation
                if annot["segmentation"] is None:
                    segm = None
                else:
                    i = 0
                    segm = []
                    while i + 1 < len(annot["segmentation"][0]):
                        segm.append((int(annot["segmentation"][0][i]), int(annot["segmentation"][0][i + 1])))
                        i += 2

                # save parsed annotation
                annots[class_reference_table[class_name]].append(
                    Annotation(class_reference_table[class_name], left, top, width, height, segm, an_type=an_type)
                )

        return FullPage((image_width, image_height), annots, class_output_names)
    
    @staticmethod
    def from_dolores_coco_file(
            file_path: Path,
            class_reference_table: dict[str, int],
            class_output_names: list[str],
            an_type: AnnotationType = AnnotationType.GROUND_TRUTH
    ) -> "FullPage":
        from ..FullPage import FullPage

        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)
        assert len(data["images"]) == 1
        
        # get image info
        image_info = data["images"][0]
        image_width, image_height = image_info["width"], image_info["height"]

        # collected annotations
        annots: list[list[Annotation]] = [[] for _ in range(len(class_output_names))]
        # class id to class names inside the loaded document
        class_id_to_name: dict[int, str] = {c["id"] : c["name"] for c in data["categories"]}
        for annotation in data["annotations"]:
            # convert loaded names to names wanted by the loading script
            class_name = class_id_to_name[annotation["categoryId"]]
            global_class_id = class_reference_table.get(class_name)
            if global_class_id is None:
                continue
            # create annotation
            left, top, width, height = annotation["bbox"]
            if (left > image_width
                or left + width > image_width
                or top > image_height
                or top + height > image_height):
                print(f"Warning: Bbox {class_name, left, top, width, height}out of bounds, file name: {image_info['file_name']}, object id: {annotation['id']}")
                # continue
            try:
                annots[global_class_id].append(
                    Annotation(global_class_id, left, top, width, height, segmentation=None, an_type=an_type)
                )
            except AssertionError as e:
                print(f"Warning: {str(e)}, file name: {image_info['file_name']}, object id: {annotation['id']}")
        
        return FullPage((image_width, image_height), annots, class_output_names)

    @staticmethod
    def save_annotation(
            page: "FullPage",
            output_path: Path
    ) -> None:
        with open(output_path, "w") as f:
            json.dump(page, f, indent=4, cls=COCOFullPageEncoder)


class COCOFullPageEncoder(JSONEncoder):
    def default(self, o):
        from ..FullPage import FullPage
        if isinstance(o, FullPage):
            output = {
                # "source": obj.source,
                "width": o.size[0],
                "height": o.size[1],
            }
            for i in range(len(o.class_names)):
                output[o.class_names[i]] = o.annotations[i] # type: ignore
            return output
        elif isinstance(o, Annotation):
            return COCOAnnotationEncoder().default(o)

        return super().default(o)


class COCOAnnotationEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, Annotation):
            # flatten
            segm = []
            for x, y in o.segmentation: # type: ignore
                segm.append(x)
                segm.append(y)

            return {
                "left": o.bbox.left,
                "top": o.bbox.top,
                "width": o.bbox.width,
                "height": o.bbox.height,
                "segmentation": [segm],
            }
        return super().default(o)
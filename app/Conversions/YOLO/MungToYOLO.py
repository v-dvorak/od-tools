import json
from pathlib import Path

from mung.io import read_nodes_from_file
from tqdm import tqdm

from .AnnotationClasses import YOLODetection, YOLOSegmentation, YOLOFullPageDetection
from .. import ConversionUtils
from ..COCO.AnnotationClasses import COCOAnnotation, COCOFullPage


def process_mung_batch_to_yolo(
        data: list[tuple[Path, Path]],
        output_paths: tuple[Path, Path],
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        mode: str = "detection",
        image_format: str = "jpg",
        resize: int = None,
):
    images_dir, annot_dir = output_paths

    for image, annot_file in tqdm(data):
        dato_name = image.stem
        image_size = ConversionUtils.get_num_pixels(image)
        # use mung to retrieve subpages from xml
        nodes = read_nodes_from_file(annot_file.__str__())

        annots = []
        # process each node
        for node in nodes:
            if node.class_name in class_reference_table:
                annots.append(COCOAnnotation.from_mung_node(class_reference_table[node.class_name], node))
        # create page class for single input data
        full_page = COCOFullPage(image_size, annots, class_output_names)

        if mode == "detection":
            # transform to yolo
            yolo_full_page = YOLOFullPageDetection.from_coco_page(full_page)

            with open(annot_dir / f"{dato_name}.txt", "w") as f:
                for annotation in yolo_full_page.annotations:
                    f.write(annotation.__str__() + "\n")
        elif mode == "segmentation":
            # transform to yolo
            pass

        # copy image
        ConversionUtils.copy_and_resize_image(
            image,
            images_dir / (dato_name + f".{image_format}"),
            max_size=resize
        )

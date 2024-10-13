import json
from pathlib import Path

from mung.io import read_nodes_from_file
from tqdm import tqdm

from .AnnotationClasses import COCOAnnotation, COCOFullPage, COCOFullPageEncoder
from .. import ConversionUtils


def process_mung_batch_to_coco(
        data: list[tuple[Path, Path]],
        output_paths: tuple[Path, Path],
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        image_format: str = "jpg",
        resize: int = None,
):
    images_dir, annot_dir = output_paths

    for image, annot_file in tqdm(data):
        dato_name = image.stem
        image_size = ConversionUtils.get_num_pixels(image)
        # use mung to retrieve annotations from xml
        nodes = read_nodes_from_file(annot_file.__str__())

        annots = []
        # process each node
        for node in nodes:
            if node.class_name in class_reference_table:
                annots.append(COCOAnnotation.from_mung_node(class_reference_table[node.class_name], node))
        # create page class for single input data
        full_page = COCOFullPage(image_size, annots, class_output_names)

        # save annotations
        with open(annot_dir / (dato_name + ".json"), "w") as f:
            json.dump(full_page, f, indent=4, cls=COCOFullPageEncoder)

        # copy image
        ConversionUtils.copy_and_resize_image(
            image,
            images_dir / (dato_name + f".{image_format}"),
            max_size=resize
        )

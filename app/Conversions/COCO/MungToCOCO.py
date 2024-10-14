import json
from pathlib import Path

import cv2
from mung.io import read_nodes_from_file
from tqdm import tqdm

from .AnnotationClasses import COCOAnnotation, COCOFullPage, COCOFullPageEncoder
from .. import ConversionUtils
from ..COCO.AnnotationClasses import COCOSplitPage
from ...Splitting import SplitUtils


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
        # use mung to retrieve subpages from xml
        nodes = read_nodes_from_file(annot_file.__str__())

        annots = []
        # process each node
        for node in nodes:
            if node.class_name in class_reference_table:
                annots.append(COCOAnnotation.from_mung_node(class_reference_table[node.class_name], node))
        # create page class for single input data
        full_page = COCOFullPage.from_list_of_coco_annotations(image_size, annots, class_output_names)

        # save subpages
        with open(annot_dir / (dato_name + ".json"), "w") as f:
            json.dump(full_page, f, indent=4, cls=COCOFullPageEncoder)

        # copy image
        ConversionUtils.copy_and_resize_image(
            image,
            images_dir / (dato_name + f".{image_format}"),
            max_size=resize
        )


def process_mung_batch_to_coco_with_splitting(
        data: list[tuple[Path, Path]],
        output_paths: tuple[Path, Path],
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        image_format: str = "jpg"
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
        full_page = COCOFullPage.from_list_of_coco_annotations(image_size, annots, class_output_names)

        splits = SplitUtils.create_split_box_matrix(image_size)

        split_page = COCOSplitPage.from_coco_full_page(full_page, splits)

        img = cv2.imread(image)

        for row in range(len(split_page.subpages)):
            for col in range(len(split_page.subpages[0])):
                # save subpage images
                index_in_page = row * len(split_page.subpages) + col

                cutout = splits[row][col]
                sub_img = img[cutout.top:cutout.bottom, cutout.left:cutout.right]

                # SplitUtils.draw_rectangles_on_image(
                #     sub_img,
                #     [SplitUtils.Rectangle.from_coco_annotation(a) for a in split_page.subpages[row][col].annotations[0]],
                #     thickness=2,
                #     loaded=True,
                #     color=(255, 0, 0),
                # )
                # TODO: separate processing and saving parts, reuse processing
                # coco -> save
                # coco -> yolo -> save

                # input("..")
                cv2.imwrite((images_dir / (dato_name + f"-{index_in_page}.{image_format}")).__str__(), sub_img)

                # save subpage annotations
                with open(annot_dir / (dato_name + f"-{index_in_page}.json"), "w") as f:
                    json.dump(split_page.subpages[row][col], f, indent=4, cls=COCOFullPageEncoder)

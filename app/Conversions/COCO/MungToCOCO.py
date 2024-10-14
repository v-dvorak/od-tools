import json
from pathlib import Path

import cv2
from mung.io import read_nodes_from_file
from tqdm import tqdm

from .AnnotationClasses import COCOAnnotation, COCOFullPage, COCOFullPageEncoder
from .. import ConversionUtils
from ..COCO.AnnotationClasses import COCOSplitPage
from ..YOLO.AnnotationClasses import YOLOFullPageDetection
from ...Splitting import SplitUtils
from ...Splitting.SplitUtils import Rectangle


def process_mung_page_to_coco(
        path_to_image: Path,
        path_to_annotation: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
) -> COCOFullPage:
    # get image size (width, height)
    image_size = ConversionUtils.get_num_pixels(path_to_image)
    # use mung to retrieve subpages from xml
    nodes = read_nodes_from_file(path_to_annotation.__str__())

    annots = []
    # process each node
    for node in nodes:
        if node.class_name in class_reference_table:
            annots.append(COCOAnnotation.from_mung_node(class_reference_table[node.class_name], node))

    # create single page
    full_page = COCOFullPage.from_list_of_coco_annotations(image_size, annots, class_output_names)
    return full_page


def process_split_batch(
        data: list[tuple[Path, Path]],
        output_paths: tuple[Path, Path],
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        image_format: str = "jpg",
        window_size: tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.25,
        mode: str = "detection",
        output_format: str = "coco",
):
    image_output_dir, annotation_output_dir = output_paths

    for path_to_image, path_to_annotations in tqdm(data):
        # load and proces page to classes
        split_page = process_mung_page_to_coco_with_split(
            path_to_image,
            path_to_annotations,
            class_reference_table,
            class_output_names,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
        )
        # save images (same for both output_format)
        save_split_image(
            path_to_image,
            image_output_dir,
            split_page.splits,
            image_format=image_format,
        )
        # save annotations
        if output_format == "coco":
            save_split_coco_annotation(
                path_to_image.stem,
                annotation_output_dir,
                split_page
            )
        elif output_format == "yolo":
            save_split_yolo_annotation(
                path_to_image.stem,
                annotation_output_dir,
                split_page,
                mode=mode,
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


def save_split_image(
        path_to_image: Path,
        output_dir: Path,
        splits: list[list[Rectangle]],
        image_format: str = "jpg",
) -> None:
    img = cv2.imread(path_to_image.__str__())
    dato_name = path_to_image.stem

    for row in range(len(splits)):
        for col in range(len(splits[0])):
            # save subpage images

            cutout = splits[row][col]
            sub_img = img[cutout.top:cutout.bottom, cutout.left:cutout.right]

            # debug viz
            # SplitUtils.draw_rectangles_on_image(
            #     sub_img,
            #     [SplitUtils.Rectangle.from_coco_annotation(a) for a in split_page.subpages[row][col].annotations[0]],
            #     thickness=2,
            #     loaded=True,
            #     color=(255, 0, 0),
            # )

            # input("..")
            cv2.imwrite((output_dir / (dato_name + f"-{row}-{col}.{image_format}")).__str__(), sub_img)


def save_split_coco_annotation(
        dato_name: str,
        output_dir: Path,
        split_page: COCOSplitPage
) -> None:
    # save subpage annotations
    for row in range(len(split_page.subpages)):
        for col in range(len(split_page.subpages[0])):
            with open(output_dir / (dato_name + f"-{row}-{col}.json"), "w") as f:
                json.dump(split_page.subpages[row][col], f, indent=4, cls=COCOFullPageEncoder)


def save_split_yolo_annotation(
        dato_name: str,
        output_dir: Path,
        split_page: COCOSplitPage,
        mode: str = "detection",
) -> None:
    for row in range(len(split_page.subpages)):
        for col in range(len(split_page.subpages[0])):
            if mode == "detection":
                yolo_fp = YOLOFullPageDetection.from_coco_page(split_page.subpages[row][col])

                with open(output_dir / f"{dato_name}-{row}-{col}.txt", "w") as f:
                    for annotation in yolo_fp.annotations:
                        f.write(annotation.__str__() + "\n")


def process_mung_page_to_coco_with_split(
        path_to_image: Path,
        path_to_annotation: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        window_size: tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.25,
) -> COCOSplitPage:
    # create single page
    full_page = process_mung_page_to_coco(
        path_to_image,
        path_to_annotation,
        class_reference_table,
        class_output_names,
    )

    # create splits
    splits = SplitUtils.create_split_box_matrix(
        full_page.size,
        window_size=window_size,
        overlap_ratio=overlap_ratio
    )

    # split page based on created splits
    split_page = COCOSplitPage.from_coco_full_page(full_page, splits)
    return split_page

    # img = cv2.imread(path_to_image.__str__())
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

            # # input("..")
            # cv2.imwrite((images_dir / (dato_name + f"-{index_in_page}.{image_format}")).__str__(), sub_img)
            #
            # # save subpage annotations
            # with open(annot_dir / (dato_name + f"-{index_in_page}.json"), "w") as f:
            #     json.dump(split_page.subpages[row][col], f, indent=4, cls=COCOFullPageEncoder)

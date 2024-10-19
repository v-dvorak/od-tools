from pathlib import Path

from mung.io import read_nodes_from_file
from tqdm import tqdm

from app.Conversions import ConversionUtils
from app.Conversions.COCO.AnnotationClasses import COCOAnnotation, COCOFullPage
from app.Conversions.COCO.AnnotationClasses import COCOSplitPage
from app.Splitting import SplitUtils
from . import DataSaving


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


def process_normal_batch(
        data: list[tuple[Path, Path]],
        output_paths: tuple[Path, Path],
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        annot_format: str = "jpg",
        mode: str = "detection",
        output_format: str = "coco",
        resize: int = None
) -> None:
    image_output_dir, annotation_output_dir = output_paths

    for path_to_image, path_to_annotations in tqdm(data):
        # load and proces page to classes
        page = process_mung_page_to_coco(
            path_to_image,
            path_to_annotations,
            class_reference_table,
            class_output_names
        )
        # save image
        ConversionUtils.copy_and_resize_image(
            path_to_image,
            image_output_dir / (path_to_image.stem + f".{annot_format}"),
            max_size=resize
        )

        # save annotations
        if output_format == "coco":
            DataSaving._save_full_page_coco_annotation(
                path_to_image.stem,
                annotation_output_dir,
                page
            )
        elif output_format == "yolo":
            DataSaving._save_full_page_yolo_annotation_from_coco_full_page(
                path_to_image.stem,
                annotation_output_dir,
                page,
                mode=mode
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


def process_split_batch(
        data: list[tuple[Path, Path]],
        output_paths: tuple[Path, Path],
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        image_format: str = "jpg",
        window_size: tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.25,
        mode: str = "detection",
        annot_format: str = "coco",
) -> None:
    image_output_dir, annotation_output_dir = output_paths

    for path_to_image, path_to_annotations in tqdm(data, position=0):
        # load and proces page to classes
        split_page = process_mung_page_to_coco_with_split(
            path_to_image,
            path_to_annotations,
            class_reference_table,
            class_output_names,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
        )
        # save images (same for both annot_format)
        DataSaving._save_split_page_image(
            path_to_image,
            image_output_dir,
            split_page.splits,
            image_format=image_format,
        )
        # save annotations
        if annot_format == "coco":
            DataSaving._save_split_page_coco_annotation(
                path_to_image.stem,
                annotation_output_dir,
                split_page
            )
        elif annot_format == "yolo":
            DataSaving._save_split_page_yolo_annotation_from_coco_split_page(
                path_to_image.stem,
                annotation_output_dir,
                split_page,
                mode=mode,
            )
        else:
            raise ValueError(f"Unsupported output format: {annot_format}")


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

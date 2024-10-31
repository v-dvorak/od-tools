from pathlib import Path

from tqdm import tqdm

from . import DataSaving
from .Formats import InputFormat, OutputFormat
from ..Conversions import ConversionUtils
from ..Conversions.COCO.AnnotationClasses import COCOFullPage
from ..Conversions.COCO.AnnotationClasses import COCOSplitPage
from ..Splitting import SplitUtils


def process_normal_batch(
        data: list[tuple[Path, Path]],
        output_paths: tuple[Path, Path],
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        input_format: InputFormat,
        output_format: OutputFormat,
        image_format: str = "jpg",
        resize: int = None
) -> None:
    image_output_dir, annotation_output_dir = output_paths

    for path_to_image, path_to_annotation in tqdm(data):
        # load and process page to classes
        page = COCOFullPage.load_from_file(
            path_to_annotation,
            path_to_image,
            class_reference_table,
            class_output_names,
            input_format
        )
        # save image
        ConversionUtils.copy_and_resize_image(
            path_to_image,
            image_output_dir / (path_to_image.stem + f".{image_format}"),
            max_size=resize
        )

        # save annotations
        page.save_to_file(
            annotation_output_dir,
            path_to_image.stem,
            output_format,
        )


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
    full_page = COCOFullPage.from_mung(
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

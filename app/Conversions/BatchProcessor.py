from pathlib import Path

import cv2
from tqdm import tqdm

from .Formats import InputFormat, OutputFormat
from ..Conversions import ConversionUtils
from ..Conversions.Annotations import FullPage, SplitPage
from ..Splitting import SplitUtils
from ..Splitting.SplitUtils import BoundingBox


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
        page = FullPage.load_from_file(
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
        input_format: InputFormat,
        output_format: OutputFormat,
        image_format: str = "jpg",
        window_size: tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.25
) -> None:
    image_output_dir, annotation_output_dir = output_paths

    for path_to_image, path_to_annotations in tqdm(data, position=0):
        # load and process page to classes
        split_page = create_splits_from_full_page(
            path_to_image,
            path_to_annotations,
            class_reference_table,
            class_output_names,
            input_format,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
        )
        # save images (same for both annot_format)
        save_split_page_image(
            path_to_image,
            image_output_dir,
            split_page.splits,
            image_format=image_format,
        )
        # save annotations
        split_page.save_to_file(
            annotation_output_dir,
            path_to_image.stem,
            output_format
        )


def create_splits_from_full_page(
        path_to_image: Path,
        path_to_annotation: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        input_format: InputFormat,
        window_size: tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.25,
) -> SplitPage:
    # create single page
    page = FullPage.load_from_file(
        path_to_annotation,
        path_to_image,
        class_reference_table,
        class_output_names,
        input_format
    )

    # create splits
    splits = SplitUtils.create_split_box_matrix(
        page.size,
        window_size=window_size,
        overlap_ratio=overlap_ratio
    )

    # split page based on created splits
    split_page = SplitPage.from_coco_full_page(page, splits)
    return split_page


def save_split_page_image(
        path_to_image: Path,
        output_dir: Path,
        splits: list[list[BoundingBox]],
        image_format: str = "jpg",
) -> None:
    """
    Loads and splits image given by path and saves it to output_dir.
    Images are stored in output_format `"name"-"row"-"column"."image_format"`.

    :param path_to_image: image to process
    :param output_dir: directory to save all split images to
    :param splits: list of bounding boxes that define splits to save
    :param image_format: output_format of image to save to
    """
    img = cv2.imread(path_to_image.__str__())
    dato_name = path_to_image.stem

    for row in range(len(splits)):
        for col in range(len(splits[0])):
            # save subpage images

            cutout = splits[row][col]
            sub_img = img[cutout.top:cutout.bottom, cutout.left:cutout.right]

            # DEBUG viz
            # SplitUtils.draw_rectangles_on_image(
            #     sub_img,
            #     [SplitUtils.BoundingBox.from_coco_annotation(a) for a in full_page.subpages[row][col].annotations[0]],
            #     thickness=2,
            #     loaded=True,
            #     color=(255, 0, 0),
            # )

            # input("..")
            cv2.imwrite((output_dir / (dato_name + f"-{row}-{col}.{image_format}")).__str__(), sub_img)

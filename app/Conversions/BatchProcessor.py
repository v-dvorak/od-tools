from pathlib import Path

import cv2
from tqdm import tqdm

from .Formats import InputFormat, OutputFormat
from .. import Splitting
from ..Conversions import ConversionUtils
from ..Conversions.Annotations import FullPage, SplitPage
from ..Conversions.BoundingBox import BoundingBox


def process_normal_batch(
        data: list[tuple[Path, Path]],
        image_out_dir: Path,
        annot_out_dir: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        input_format: InputFormat,
        output_format: OutputFormat,
        image_format: str = "jpg",
        resize: int = None
) -> None:
    for image, annotation in tqdm(data):
        # load and process page to classes
        page = FullPage.load_from_file(
            annotation,
            image,
            class_reference_table,
            class_output_names,
            input_format
        )
        # save image
        ConversionUtils.copy_and_resize_image(
            image,
            image_out_dir / (image.stem + f".{image_format}"),
            max_size=resize
        )

        # save annotations
        page.save_to_file(
            annot_out_dir,
            image.stem,
            output_format,
        )


def _process_single_split_image(
        path_to_image: Path,
        path_to_annot: Path,
        image_out_dir: Path,
        annot_out_dir: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        input_format: InputFormat,
        output_format: OutputFormat,
        image_format: str = "jpg",
        window_size: tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.25,
        name_add_tags: str = None,
):
    split_page = create_splits_from_full_page(
        path_to_image,
        path_to_annot,
        class_reference_table,
        class_output_names,
        input_format,
        window_size=window_size,
        overlap_ratio=overlap_ratio,
    )
    # save images (same for both annot_format)
    save_split_page_image(
        path_to_image,
        image_out_dir,
        split_page.splits,
        image_format=image_format,
        add_tags=name_add_tags,
    )

    # save annotations
    split_page.save_to_file(
        annot_out_dir,
        path_to_image.stem,
        output_format,
        add_tags=name_add_tags,
    )


def process_split_batch(
        data: list[tuple[Path, Path]],
        image_out_dir: Path,
        annot_out_dir: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        input_format: InputFormat,
        output_format: OutputFormat,
        image_format: str = "jpg",
        window_size: tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.25,
        aug_ratios: list[float] = None,
) -> None:
    if aug_ratios is None:
        for path_to_image, path_to_annot in tqdm(data, position=0):
            # load and process page to classes
            _process_single_split_image(
                path_to_image,
                path_to_annot,
                image_out_dir,
                annot_out_dir,
                class_reference_table,
                class_output_names,
                input_format,
                output_format,
                image_format=image_format,
                window_size=window_size,
                overlap_ratio=overlap_ratio
            )
    else:
        # create tags for given augmentation ratios
        aug_tags = []
        for aug_ratio in aug_ratios:
            aug_tags.append(f"x{aug_ratio:.2f}".replace(".", ""))

        for path_to_image, path_to_annot in tqdm(data, position=0):

            for aug_ratio, aug_tag in zip(aug_ratios, aug_tags):
                # computer size of a new augmented window
                new_ws = (int(window_size[0] * aug_ratio), int(window_size[1] * aug_ratio))
                # load and process page to classes
                _process_single_split_image(
                    path_to_image,
                    path_to_annot,
                    image_out_dir,
                    annot_out_dir,
                    class_reference_table,
                    class_output_names,
                    input_format,
                    output_format,
                    image_format=image_format,
                    window_size=new_ws,
                    overlap_ratio=overlap_ratio,
                    name_add_tags=aug_tag
                )


def create_splits_from_full_page(
        image_path: Path,
        annot_path: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        input_format: InputFormat,
        window_size: tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.25,
) -> SplitPage:
    # create single page
    page = FullPage.load_from_file(
        annot_path,
        image_path,
        class_reference_table,
        class_output_names,
        input_format
    )

    # create splits
    splits = Splitting.create_split_box_matrix(
        page.size,
        window_size=window_size,
        overlap_ratio=overlap_ratio
    )

    # split page based on created splits
    return SplitPage.from_full_page(page, splits)


def _create_image_name(dato_name: str, row: int, col: int, image_format: str, add_tags: str = None,
                       sep: str = "-") -> str:
    return (
        sep.join([dato_name, str(row), str(col)]) if add_tags is None
        else sep.join([dato_name, str(row), str(col), add_tags])
    ) + f".{image_format}"


def save_split_page_image(
        path_to_image: Path,
        output_dir: Path,
        splits: list[list[BoundingBox]],
        image_format: str = "jpg",
        add_tags: str = None
) -> None:
    """
    Loads and splits image given by path and saves it to output_dir.
    Images are stored in output_format `"name"-"row"-"column"."image_format"`.

    :param path_to_image: image to process
    :param output_dir: directory to save all split images to
    :param splits: list of bounding boxes that define splits to save
    :param image_format: output_format of image to save to
    :param add_tags: additional tags to add when saving image
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
            file_name = _create_image_name(dato_name, row, col, image_format, add_tags=add_tags)
            cv2.imwrite((output_dir / file_name).__str__(), sub_img)

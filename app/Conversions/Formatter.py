import warnings
from pathlib import Path

from . import ConversionUtils
from . import MungToCOCO


def format_dataset(
        images_path: Path,
        annotations_path: Path,
        dataset_path: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        output_format: str = "coco",
        mode: str = None,
        split_ratio: float = 0.9,
        resize: int = None,
        seed: int = 42,
        image_format: str = "jpg",
        window_size: tuple[int, int] = (640, 640),
        overlap_ratio: float = 0.25,
        image_splitting: bool = False,
        verbose: bool = False,
) -> None:
    """
    Finds all images and subpages inside given files
    and processes them according to the split ratio and output annot_format.

    :param images_path: path to directory with images
    :param annotations_path: path to directory with labels
    :param dataset_path: path to final dataset, all data will be outputted here

    :param class_reference_table: dictionary, a function that assigns class id by class name
    :param class_output_names: list of class names

    :param output_format: "coco" or "yolo", defines output annot_format
    :param mode: detection or segmentation
    :param split_ratio: train/test split ratio

    :param resize: resizes images so that the longer side is this many pixels long
    :param seed: seed for dataset shuffling
    :param image_format: annot_format in which the images are saved

    :param window_size: size of the sliding window applied to image in case of image splitting
    :param overlap_ratio: overlap ratio between two tiles in case of image splitting
    :param image_splitting: whether to split images according to the split ratio and sliding window

    :param verbose: make script verbose
    """
    # check parameters
    if mode is not None and output_format == "coco":
        warnings.warn("COCO annot_format exports in both modes (\"detection\", \"segmentation\")at the same time.")
    if mode is not None and mode not in ["detection", "segmentation"]:
        raise ValueError("mode must be either \"detection\" or \"segmentation\"")
    if mode is None:
        mode = "detection"

    if output_format not in ["coco", "yolo"]:
        raise ValueError("annot_format must be either \"coco\" or \"yolo\"")

    # load data from given paths
    images = sorted(list(images_path.rglob(f"*.{image_format}")))
    annotations = sorted(list(annotations_path.rglob(f"*.xml")))
    data = list(zip(images, annotations))

    # dump everything into one directory
    if split_ratio == 1.0:
        # set up folders
        images_dir = dataset_path / "images"
        annot_dir = dataset_path / "labels"

        images_dir.mkdir(exist_ok=True, parents=True)
        annot_dir.mkdir(exist_ok=True, parents=True)

        if image_splitting:
            MungToCOCO.process_split_batch(
                data,
                (images_dir, annot_dir),
                class_reference_table,
                class_output_names,
                image_format=image_format,
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                mode=mode,
                annot_format=output_format
            )
        else:
            MungToCOCO.process_normal_batch(
                data,
                (images_dir, annot_dir),
                class_reference_table,
                class_output_names,
                annot_format=image_format,
                mode=mode,
                output_format=output_format,
                resize=resize
            )

    # split to train/test
    else:
        # set up folders
        train_image_dir = dataset_path / "images" / "train"
        val_image_dir = dataset_path / "images" / "val"
        train_annotation_dir = dataset_path / "labels" / "train"
        val_annotation_dir = dataset_path / "labels" / "val"

        train_image_dir.mkdir(exist_ok=True, parents=True)
        val_image_dir.mkdir(exist_ok=True, parents=True)
        train_annotation_dir.mkdir(exist_ok=True, parents=True)
        val_annotation_dir.mkdir(exist_ok=True, parents=True)

        # split
        train_data, val_data = ConversionUtils.split_dataset(data, split_ratio=split_ratio, seed=seed)

        if image_splitting:
            MungToCOCO.process_split_batch(
                train_data,
                (train_image_dir, train_annotation_dir),
                class_reference_table,
                class_output_names,
                image_format=image_format,
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                mode=mode,
                annot_format=output_format
            )

            MungToCOCO.process_split_batch(
                val_data,
                (val_image_dir, val_annotation_dir),
                class_reference_table,
                class_output_names,
                image_format=image_format,
                window_size=window_size,
                overlap_ratio=overlap_ratio,
                mode=mode,
                annot_format=output_format
            )
        else:
            MungToCOCO.process_normal_batch(
                train_data,
                (train_image_dir, train_annotation_dir),
                class_reference_table,
                class_output_names,
                annot_format=image_format,
                mode=mode,
                output_format=output_format,
                resize=resize
            )

            MungToCOCO.process_normal_batch(
                val_data,
                (val_image_dir, val_annotation_dir),
                class_reference_table,
                class_output_names,
                annot_format=image_format,
                mode=mode,
                output_format=output_format,
                resize=resize
            )

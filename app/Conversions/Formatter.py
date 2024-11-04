from pathlib import Path

import yaml

from . import BatchProcessor
from . import ConversionUtils
from .Formats import InputFormat, OutputFormat


def format_dataset(
        images_path: Path,
        annotations_path: Path,
        dataset_path: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        input_format: InputFormat,
        output_format: OutputFormat,
        image_format: str = "jpg",
        split_ratio: float = 0.9,
        resize: int = None,
        seed: int = 42,
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

    :param input_format: defines input output_format
    :param output_format: defines output annot_format

    :param split_ratio: train/test split ratio
    :param resize: resizes images so that the longer side is this many pixels long
    :param seed: seed for dataset shuffling
    :param image_format: annot_format in which the images are saved

    :param window_size: size of the sliding window applied to image in case of image splitting
    :param overlap_ratio: overlap ratio between two tiles in case of image splitting
    :param image_splitting: whether to split images according to the split ratio and sliding window

    :param verbose: make script verbose
    """
    # load data from given paths
    images = sorted(list(images_path.rglob(f"*.{image_format}")))
    annotations = sorted(list(annotations_path.rglob(f"*.{input_format.to_annotation_extension()}")))
    data = list(zip(images, annotations))

    # dump everything into one directory
    if split_ratio == 1.0:
        # set up folders
        images_dir = dataset_path / "images"
        annot_dir = dataset_path / "labels"

        images_dir.mkdir(exist_ok=True, parents=True)
        annot_dir.mkdir(exist_ok=True, parents=True)

        if image_splitting:
            BatchProcessor.process_split_batch(
                data,
                (images_dir, annot_dir),
                class_reference_table,
                class_output_names,
                input_format,
                output_format,
                image_format=image_format,
                window_size=window_size,
                overlap_ratio=overlap_ratio
            )
        else:
            BatchProcessor.process_normal_batch(
                data,
                (images_dir, annot_dir),
                class_reference_table,
                class_output_names,
                input_format,
                output_format,
                image_format=image_format,
                resize=resize
            )

        if output_format == OutputFormat.YOLO_DETECTION or output_format == OutputFormat.YOLO_SEGMENTATION:
            _create_yaml_config_for_yolo(
                dataset_path,
                images_dir,
                class_output_names,
                verbose=verbose
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
            BatchProcessor.process_split_batch(
                train_data,
                (train_image_dir, train_annotation_dir),
                class_reference_table,
                class_output_names,
                input_format,
                output_format,
                image_format=image_format,
                window_size=window_size,
                overlap_ratio=overlap_ratio
            )

            BatchProcessor.process_split_batch(
                val_data,
                (val_image_dir, val_annotation_dir),
                class_reference_table,
                class_output_names,
                input_format,
                output_format,
                image_format=image_format,
                window_size=window_size,
                overlap_ratio=overlap_ratio
            )
        else:
            BatchProcessor.process_normal_batch(
                train_data,
                (train_image_dir, train_annotation_dir),
                class_reference_table,
                class_output_names,
                input_format,
                output_format,
                image_format=image_format,
                resize=resize
            )

            BatchProcessor.process_normal_batch(
                val_data,
                (val_image_dir, val_annotation_dir),
                class_reference_table,
                class_output_names,
                input_format,
                output_format,
                image_format=image_format,
                resize=resize
            )

        if output_format == OutputFormat.YOLO_DETECTION or output_format == OutputFormat.YOLO_SEGMENTATION:
            _create_yaml_config_for_yolo(
                dataset_path,
                train_image_dir,
                class_output_names,
                verbose=verbose,
                val_path=val_image_dir
            )


def _create_yaml_config_for_yolo(
        dataset_path: Path,
        train_path: Path,
        class_output_names: list[str],
        val_path: Path = None,
        config_name: str = "config",
        verbose: bool = False,
):
    """
    Creates .yaml file in YOLO output_format necessary for model training.

    :param dataset_path: path to dataset directory
    :param train_path: path to train directory
    :param class_output_names: names of classes
    :param val_path: path to validation directory, optional, if not provided, defaults to `train_path`
    :param config_name: name of config file, default is "config"
    """
    if val_path is None:
        val_path = train_path

    names = {}
    for i, class_name in enumerate(class_output_names):
        names[i] = class_name

    data = {
        "path": str(dataset_path.absolute().resolve()),
        "train": str(train_path.absolute().resolve()),
        "val": str(val_path.absolute().resolve()),
        "names": names,
    }

    with open(dataset_path / f"{config_name}.yaml", "w") as f:
        yaml.dump(data, f, sort_keys=False, indent=4)

    if verbose:
        print(f"Created {config_name}.yaml in {dataset_path.absolute().resolve()}.")

import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

from . import BatchProcessor
from . import ConversionUtils
from .Formats import InputFormat, OutputFormat


def _setup_split_dirs(
        output_dir: Path,
        exist_ok: bool = True,
        parents: bool = True
) -> tuple[Path, Path, Path, Path]:
    train_image_dir = output_dir / "images" / "train"
    val_image_dir = output_dir / "images" / "val"
    train_annot_dir = output_dir / "labels" / "train"
    val_annot_dir = output_dir / "labels" / "val"

    train_image_dir.mkdir(exist_ok=exist_ok, parents=parents)
    val_image_dir.mkdir(exist_ok=exist_ok, parents=parents)
    train_annot_dir.mkdir(exist_ok=exist_ok, parents=parents)
    val_annot_dir.mkdir(exist_ok=exist_ok, parents=parents)

    return train_image_dir, val_image_dir, train_annot_dir, val_annot_dir


def _setup_no_split_dirs(
        output_dir: Path,
        exist_ok: bool = True,
        parents: bool = True
) -> tuple[Path, Path]:
    images_dir = output_dir / "images"
    annot_dir = output_dir / "labels"

    images_dir.mkdir(exist_ok=exist_ok, parents=parents)
    annot_dir.mkdir(exist_ok=exist_ok, parents=parents)

    return images_dir, annot_dir


def _load_data_from_paths(
        images_dir: Path,
        images_search_regex: str,
        annotations_dir: Path,
        annotations_search_regex: str,
) -> list[tuple[Path, Path]]:
    images = sorted(list(images_dir.rglob(images_search_regex)))
    annotations = sorted(list(annotations_dir.rglob(annotations_search_regex)))
    return list(zip(images, annotations))


def split_and_save_dataset(
        output_path: Path,
        image_source_dir: Path,
        annot_source_dir: Path,
        split_ratio: float = 0.9,
        seed: int = 42,
        verbose: bool = False,
) -> None:
    """
    Splits images and annotations into train and test (val) datasets.
    Given data are split based on the split_ratio regardless of their format.
    Config file will not be created.

    :param output_path: path to save split dataset
    :param image_source_dir: path to image directory
    :param annot_source_dir: path to annotation directory

    :param split_ratio: split ratio
    :param seed: random seed

    :param verbose: make script verbose
    """
    data = _load_data_from_paths(
        image_source_dir, "*",
        annot_source_dir, "*"
    )
    train_image_dir, val_image_dir, train_annot_dir, val_annot_dir = _setup_split_dirs(output_path)

    # split
    train_data, val_data = ConversionUtils.split_dataset(data, split_ratio=split_ratio, seed=seed)

    for image, annotation in tqdm(train_data, desc="Processing train", disable=not verbose):
        shutil.copy(image, train_image_dir / image.name)
        shutil.copy(annotation, train_annot_dir / annotation.name)

    for image, annotation in tqdm(val_data, desc="Processing val", disable=not verbose):
        shutil.copy(image, val_image_dir / image.name)
        shutil.copy(annotation, val_annot_dir / annotation.name)


def format_dataset(
        output_dir: Path,
        image_source_dir: Path,
        annot_source_dir: Path,
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
    and processes them according to the split ratio and output format.

    :param image_source_dir: path to directory with images
    :param annot_source_dir: path to directory with labels
    :param output_dir: path to final dataset, all data will be outputted here

    :param class_reference_table: dictionary, a function that assigns class id by class name
    :param class_output_names: list of class names

    :param input_format: defines input output_format
    :param output_format: defines output annot_format

    :param split_ratio: train/test split ratio
    :param resize: resizes images so that the longer side is this many pixels long
    :param seed: seed for dataset shuffling
    :param image_format: format in which the images are saved

    :param window_size: size of the sliding window applied to image in case of image splitting
    :param overlap_ratio: overlap ratio between two tiles in case of image splitting
    :param image_splitting: whether to split images according to the split ratio and sliding window

    :param verbose: make script verbose
    """
    data = _load_data_from_paths(
        image_source_dir, f"*.{image_format}",
        annot_source_dir, f"*.{input_format.to_annotation_extension()}"
    )

    # dump everything into one directory
    if split_ratio == 1.0:
        # set up folders
        out_image_dir, out_annot_dir = _setup_no_split_dirs(output_dir)

        if image_splitting:
            BatchProcessor.process_split_batch(
                data,
                out_image_dir,
                out_annot_dir,
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
                out_image_dir,
                out_annot_dir,
                class_reference_table,
                class_output_names,
                input_format,
                output_format,
                image_format=image_format,
                resize=resize
            )

        if output_format == OutputFormat.YOLO_DETECTION or output_format == OutputFormat.YOLO_SEGMENTATION:
            _create_yaml_config_for_yolo(
                output_dir,
                out_image_dir,
                class_output_names,
                verbose=verbose
            )

    # split to train/test
    else:
        # setup folders
        train_image_dir, val_image_dir, train_annot_dir, val_annot_dir = _setup_split_dirs(output_dir)

        # split
        train_data, val_data = ConversionUtils.split_dataset(data, split_ratio=split_ratio, seed=seed)

        if image_splitting:
            BatchProcessor.process_split_batch(
                train_data,
                train_image_dir,
                train_annot_dir,
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
                val_image_dir,
                val_annot_dir,
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
                train_image_dir,
                train_annot_dir,
                class_reference_table,
                class_output_names,
                input_format,
                output_format,
                image_format=image_format,
                resize=resize
            )

            BatchProcessor.process_normal_batch(
                val_data,
                val_image_dir,
                val_annot_dir,
                class_reference_table,
                class_output_names,
                input_format,
                output_format,
                image_format=image_format,
                resize=resize
            )

        if output_format == OutputFormat.YOLO_DETECTION or output_format == OutputFormat.YOLO_SEGMENTATION:
            _create_yaml_config_for_yolo(
                output_dir,
                train_image_dir,
                class_output_names,
                verbose=verbose,
                val_path=val_image_dir
            )


def _create_yaml_config_for_yolo(
        output_path: Path,
        train_path: Path,
        class_output_names: list[str],
        val_path: Path = None,
        config_name: str = "config",
        verbose: bool = False,
):
    """
    Creates .yaml file in YOLO format necessary for model training.

    :param output_path: path to dataset directory
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
        "path": str(output_path.absolute().resolve()),
        "train": str(train_path.absolute().resolve()),
        "val": str(val_path.absolute().resolve()),
        "names": names,
    }

    with open(output_path / f"{config_name}.yaml", "w") as f:
        yaml.dump(data, f, sort_keys=False, indent=4)

    if verbose:
        print(f"Created {config_name}.yaml in {output_path.absolute().resolve()}.")

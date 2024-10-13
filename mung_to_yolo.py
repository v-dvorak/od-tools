from pathlib import Path

from mung.io import read_nodes_from_file
from tqdm import tqdm

from resize_images import copy_and_resize_image
from utils import mung_to_yolo_segm, mung_to_yolo_annot, split_dataset, get_num_pixels


def _process_batch_of_data(
        data: list[tuple[Path, Path]],
        output_paths: tuple[Path, Path],
        classes: dict[str, int],
        mode: str = "detection",
        resize: int = None,
        image_format: str = "jpg"
) -> None:
    """
    Processes single batch of files. Does not care about split and other higher level settings.
    """
    image_output, label_output = output_paths
    for image, annot in tqdm(data):
        dato_name = image.stem
        image_size = get_num_pixels(image)
        # use mung to retrieve annotations from xml
        nodes = read_nodes_from_file(annot.__str__())

        annots = []
        for node in nodes:
            if node.class_name in classes:
                # DETECTION
                if mode == "detection":
                    label = mung_to_yolo_annot(node, image_size)
                    label.cls = classes[node.class_name]
                    annots.append(label)
                # SEGMENTATION
                elif mode == "segmentation":
                    label = mung_to_yolo_segm(node, image_size)
                    label.cls = classes[node.class_name]
                    annots.append(label)
                else:
                    raise ValueError(f"Unknown mode {mode}")

        # save annotations
        with open(label_output / (dato_name + ".txt"), "w") as f:
            f.writelines([annot.__str__() + "\n" for annot in annots])

        # copy (and resize) image
        copy_and_resize_image(image, image_output / (dato_name + f".{image_format}"), max_size=resize)


def process_dataset(
        images_path: Path,
        annotations_path: Path,
        dataset_path: Path,
        classes: dict[str, int],
        mode: str = "detection",
        split_ratio: float = 0.9,
        resize: int = None,
        seed: int = 42,
        image_format: str = "jpg"
) -> None:
    """
    Finds all images and annotations inside given files and processes them according to the split ratio.

    :param images_path: path to directory with images
    :param annotations_path: path to directory with labels
    :param dataset_path: path to final dataset, all data will be outputted here
    :param classes: dictionary, a function that assigns class id by class name
    :param mode: detection or segmentation
    :param image_format: format in which the images are saved
    :param split_ratio: train/test split ratio
    :param resize: resizes images so that the longer side is this many pixels long
    :param seed: seed for dataset shuffling
    """
    images = sorted(list(images_path.rglob(f"*.{image_format}")))
    annotations = sorted(list(annotations_path.rglob(f"*.xml")))
    data = zip(images, annotations)

    # only create training data
    if split_ratio == 1.0:
        # set up folders
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        images_dir.mkdir(exist_ok=True, parents=True)
        labels_dir.mkdir(exist_ok=True, parents=True)

        # process
        _process_batch_of_data(
            data,
            (images_dir, labels_dir),
            classes=classes,
            mode=mode,
            image_format=image_format,
            resize=resize
        )

    # create split
    else:
        # set up folders
        train_images_dir = dataset_path / "images" / "train"
        val_images_dir = dataset_path / "images" / "val"
        train_labels_dir = dataset_path / "labels" / "train"
        val_labels_dir = dataset_path / "labels" / "val"

        train_images_dir.mkdir(exist_ok=True, parents=True)
        val_images_dir.mkdir(exist_ok=True, parents=True)
        train_labels_dir.mkdir(exist_ok=True, parents=True)
        val_labels_dir.mkdir(exist_ok=True, parents=True)

        train_data, val_data = split_dataset(data, split_ratio=split_ratio, seed=seed)

        # train
        _process_batch_of_data(
            train_data,
            (train_images_dir, train_labels_dir),
            classes=classes,
            mode=mode,
            image_format=image_format,
            resize=resize
        )

        # test
        _process_batch_of_data(
            data,
            (val_images_dir, val_labels_dir),
            classes=classes,
            mode=mode,
            image_format=image_format,
            resize=resize
        )

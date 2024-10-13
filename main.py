import argparse
import json
from pathlib import Path

from mung.io import read_nodes_from_file
from tqdm import tqdm

from annotation_classes import COCOFullPage, COCOAnnotation, COCOFullPageEncoder
from resize_images import copy_and_resize_image
from utils import get_num_pixels


def process_mung_dataset_to_coco(
        images_path: Path,
        annotations_path: Path,
        dataset_path: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        image_format: str = "jpg"
):
    # load data names
    images = sorted(list(images_path.rglob(f"*.{image_format}")))
    annotations = sorted(list(annotations_path.rglob(f"*.xml")))

    # setup folders
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
    images_dir.mkdir(exist_ok=True, parents=True)
    labels_dir.mkdir(exist_ok=True, parents=True)

    for image, annot in tqdm(zip(images, annotations)):
        dato_name = image.stem
        image_size = get_num_pixels(image)
        # use mung to retrieve annotations from xml
        nodes = read_nodes_from_file(annot.__str__())

        annots = []
        # process each node
        for node in nodes:
            if node.class_name in class_reference_table:
                annots.append(COCOAnnotation.from_mung_node(class_reference_table[node.class_name], node))
        # create page class for single input data
        full_page = COCOFullPage(image_size, annots, class_output_names)

        # save annotations
        with open(labels_dir / (dato_name + ".json"), "w") as f:
            json.dump(full_page, f, indent=4, cls=COCOFullPageEncoder)

        # only copy
        copy_and_resize_image(image, images_dir / (dato_name + f".{image_format}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Notehead experiments",
    )
    parser.add_argument("output", help="Transformed dataset destination.")
    parser.add_argument("images_path", help="Path to images.")
    parser.add_argument("annot_path", help="Path to annotations.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Make script verbose")
    parser.add_argument("-m", "--mode", default="detection",
                        help="Output dataset mode, detection or segmentation, default is \"detection\".")
    parser.add_argument("-s", "--split", type=float, default=1.0, help="Train/test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for dataset shuffling.")
    parser.add_argument("--settings", default="default_settings.json",
                        help="Path to definitions \"class name -> class id\".")
    parser.add_argument("--resize", type=int, default=None,
                        help="Resizes images so that the longer side is this many pixels long.")

    args = parser.parse_args()

    # load settings
    with open(args.settings, "r", encoding="utf8") as f:
        loaded_settings = json.load(f)

    process_mung_dataset_to_coco(
        Path(args.images_path),
        Path(args.annot_path),
        Path(args.output),
        class_reference_table=loaded_settings["class_id_reference_table"],
        class_output_names=loaded_settings["class_output_names"],
    )

    quit()

    process_dataset(
        Path(args.images_path),
        Path(args.annot_path),
        Path(args.output),
        loaded_settings["class_id_reference_table"],
        split_ratio=args.split,
        seed=args.seed,
        mode=args.mode,
        resize=args.resize,
    )

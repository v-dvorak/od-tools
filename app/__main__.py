import argparse
import json
from importlib import resources as impresources
from pathlib import Path

from . import data
from .Conversions import ConversionUtils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Notehead experiments",
    )
    parser.add_argument("output", help="Transformed dataset destination.")
    parser.add_argument("images_path", help="Path to images.")
    parser.add_argument("annot_path", help="Path to subpages.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Make script verbose")
    parser.add_argument("-m", "--mode", default=None,
                        help="Output dataset mode, detection or segmentation, default is \"detection\".")
    parser.add_argument("-s", "--split", type=float, default=1.0, help="Train/test split ratio.")
    parser.add_argument("-f", "--format", type=str, default="coco",
                        help="Output format, coco/yolo, default is coco.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for dataset shuffling.")
    parser.add_argument("--config", default=None,
                        help="Path to config, see \"default_config.json\" for example.")
    parser.add_argument("--resize", type=int, default=None,
                        help="Resizes images so that the longer side is this many pixels long.")
    parser.add_argument("--image_splitting", action="store_true", help="Split images into smaller ones.")

    args = parser.parse_args()

    # load config
    if args.config is None:
        inp_file = impresources.files(data) / "default_config.json"
        with inp_file.open("rt") as f:
            loaded_config = json.load(f)
    else:
        with open(args.settings, "r", encoding="utf8") as f:
            loaded_config = json.load(f)

    ConversionUtils.format_dataset(
        # directories
        Path(args.images_path),
        Path(args.annot_path),
        Path(args.output),
        # class ids etc.
        class_reference_table=loaded_config["class_id_reference_table"],
        class_output_names=loaded_config["class_output_names"],
        # formatting
        output_format=args.format,
        mode=args.mode,
        split_ratio=args.split,
        resize=args.resize,
        # image splitting settings
        window_size=tuple(loaded_config["window_size"]),
        overlap_ratio=loaded_config["overlap_ratio"],
        image_splitting=args.image_splitting,
        # others
        seed=args.seed,
        verbose=args.verbose
    )

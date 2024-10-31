import argparse
import json
from importlib import resources as impresources
from pathlib import Path

from . import data
from .Conversions import Formatter
from .Stats import Plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Object Detection Tools"
    )

    parser.add_argument("--config", default=None,
                        help="Path to config, see \"default_config.json\" for example.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Make script verbose")

    subparsers = parser.add_subparsers(dest="command", help="Jobs")

    # DATASET FORMATTING AND SPLITTING
    proc_parser = subparsers.add_parser("proc")
    proc_parser.add_argument("output", help="Transformed dataset destination.")
    proc_parser.add_argument("images_path", help="Path to images.")
    proc_parser.add_argument("annot_path", help="Path to subpages.")

    proc_parser.add_argument("-i", "--input_format", default="mung", choices=["mung", "coco", "yolod", "yolos"])
    proc_parser.add_argument("-o", "--output_format", default="coco", choices=["mung", "coco", "yolod", "yolos"])
    proc_parser.add_argument("--image_format", default="jpg", help="Input image format.")

    proc_parser.add_argument("-s", "--split", type=float, default=1.0, help="Train/test split ratio.")

    proc_parser.add_argument("--seed", type=int, default=42, help="Seed for dataset shuffling.")
    proc_parser.add_argument("--config", default=None,
                             help="Path to config, see \"default_config.json\" for example.")
    proc_parser.add_argument("--resize", type=int, default=None,
                             help="Resizes images so that the longer side is this many pixels long.")
    proc_parser.add_argument("--image_splitting", action="store_true", help="Split images into smaller ones.")

    # DATASET STATISTICS
    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("images_path", help="Path to images.")
    stats_parser.add_argument("annot_path", help="Path to subpages.")

    stats_parser.add_argument("-o", "--output", type=str, default=None, help="If used, plots will be saved here.")

    args = parser.parse_args()

    # load config
    if args.config is None:
        inp_file = impresources.files(data) / "default_config.json"
        with inp_file.open("rt") as f:
            loaded_config = json.load(f)
    else:
        with open(args.settings, "r", encoding="utf8") as f:
            loaded_config = json.load(f)

    if args.command == "proc":
        input_f = Formatter.InputFormat.from_string(args.input_format)
        output_f = Formatter.OutputFormat.from_string(args.output_format)

        print(input_f.name, output_f.name)
        print(input_f.to_annotation_extension())
        # quit()

        Formatter.format_dataset(
            # directories
            Path(args.images_path),
            Path(args.annot_path),
            Path(args.output),
            # class ids etc.
            class_reference_table=loaded_config["class_id_reference_table"],
            class_output_names=loaded_config["class_output_names"],
            # formatting
            input_format=input_f,
            output_format=output_f,
            split_ratio=args.split,
            resize=args.resize,
            image_format=args.image_format,
            # image splitting settings
            window_size=tuple(loaded_config["window_size"]),
            overlap_ratio=loaded_config["overlap_ratio"],
            image_splitting=args.image_splitting,
            # others
            seed=args.seed,
            verbose=args.verbose
        )
    elif args.command == "stats":
        Plots.load_and_plot_stats(
            # directories
            Path(args.images_path),
            Path(args.annot_path),
            # class ids etc.
            class_reference_table=loaded_config["class_id_reference_table"],
            class_output_names=loaded_config["class_output_names"],
            image_format="jpg",
            # others
            output_path=Path(args.output) if args.output is not None else None,
            verbose=args.verbose
        )

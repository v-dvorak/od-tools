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
    subparsers = parser.add_subparsers(dest="command", help="Jobs")

    # DATASET FORMATTING AND SPLITTING
    form_parser = subparsers.add_parser("form")
    form_parser.add_argument("output", help="Transformed dataset destination.")
    form_parser.add_argument("images_path", help="Path to images.")
    form_parser.add_argument("annot_path", help="Path to subpages.")

    form_parser.add_argument("-i", "--input_format", default="mung", choices=["mung", "coco", "yolod", "yolos"])
    form_parser.add_argument("-o", "--output_format", default="coco", choices=["mung", "coco", "yolod", "yolos"])
    form_parser.add_argument("--image_format", default="jpg", help="Input image format.")

    form_parser.add_argument("-s", "--split", type=float, default=1.0, help="Train/test split ratio.")

    form_parser.add_argument("--seed", type=int, default=42, help="Seed for dataset shuffling.")
    form_parser.add_argument("--resize", type=int, default=None,
                             help="Resizes images so that the longer side is this many pixels long.")
    form_parser.add_argument("--image_splitting", action="store_true", help="Split images into smaller ones.")

    # global arguments
    form_parser.add_argument("-v", "--verbose", action="store_true", help="Make script verbose")
    form_parser.add_argument("--config", default=None,
                             help="Path to config, see \"default_config.json\" for example.")

    # DATASET STATISTICS
    stats_parser = subparsers.add_parser("stats")
    stats_parser.add_argument("images_path", help="Path to images.")
    stats_parser.add_argument("annot_path", help="Path to subpages.")

    stats_parser.add_argument("-o", "--output", type=str, default=None, help="If used, plots will be saved here.")

    # global arguments
    stats_parser.add_argument("-v", "--verbose", action="store_true", help="Make script verbose")
    stats_parser.add_argument("--config", default=None,
                              help="Path to config, see \"default_config.json\" for example.")

    # MODEL VALIDATION
    val_parser = subparsers.add_parser("val")

    val_parser.add_argument("model_path", type=str, help="Path to model.")
    val_parser.add_argument("images_path", help="Path to images.")
    val_parser.add_argument("annot_path", help="Path to subpages.")

    val_parser.add_argument("-i", "--input_format", default="yolod", choices=["mung", "coco", "yolod", "yolos"],
                            help="Validation dataset annotation format.")
    val_parser.add_argument("-m", "--model_type", default="yolod", choices=["yolod", "yolos"],
                            help="Type of model.")

    val_parser.add_argument("--image_format", default="jpg", help="Input image format.")

    val_parser.add_argument("-o", "--overlap", type=int, help="Overlap ratio for image splits.")
    val_parser.add_argument("-c", "--count", type=int, help="How many images the model will be tested on.")
    val_parser.add_argument("-s", "--seed", type=int, default=42, help="Seed for dataset shuffling.")

    # global arguments
    val_parser.add_argument("-v", "--verbose", action="store_true", help="Make script verbose")
    val_parser.add_argument("--config", default=None,
                            help="Path to config, see \"default_config.json\" for example.")

    args = parser.parse_args()

    # load config
    if args.config is None:
        inp_file = impresources.files(data) / "default_config.json"
        with inp_file.open("rt") as f:
            loaded_config = json.load(f)
    else:
        with open(args.config, "r", encoding="utf8") as f:
            loaded_config = json.load(f)

    if args.command == "form":
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

    elif args.command == "val":
        from .Val import EvalJob, FScores
        from .Val.EvalJob import ValBoundingBox

        CLASSES = loaded_config["class_output_names"]
        # this is just to force this into loader methods without changing anything about / merging the loaded classes
        class_reference_table = {}
        for i, class_name in enumerate(CLASSES):
            class_reference_table[class_name] = i

        GROUND_TRUTH, PREDICTIONS = EvalJob.validate_model(
            Path(args.model_path),
            Path(args.images_path),
            Path(args.annot_path),
            Formatter.InputFormat.from_string(args.input_format),
            EvalJob.ModelType.YOLO_DETECTION,
            CLASSES,
            class_reference_table,
            count=args.count,
            verbose=args.verbose,
            debug=False,
        )

        GROUND_TRUTH: list[ValBoundingBox]
        PREDICTIONS: list[ValBoundingBox]
        global_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

        scores = FScores.collect_f_scores(
            GROUND_TRUTH,
            PREDICTIONS,
            CLASSES,
            iou_thresholds=global_thresholds,
            verbose=args.verbose,
        )

        FScores.plot_f_scores(global_thresholds, scores, CLASSES + ["all"])

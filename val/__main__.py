from __future__ import annotations

import argparse
from pathlib import Path

from odmetrics.bounding_box import ValBoundingBox
from . import EvalJob
from . import FScores

# CLASSES = ["system_measures", "stave_measures", "staves", "systems", "grand_staff"]
CLASSES = ["noteheadFull", "noteheadHalf"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to model.")
    parser.add_argument("images_path", help="Path to images.")
    parser.add_argument("annot_path", help="Path to subpages.")

    parser.add_argument("-o", "--overlap", type=int, help="Overlap ratio for image splits.")
    parser.add_argument("--config", default=None,
                        help="Path to config, see \"default_config.json\" for example.")
    parser.add_argument("-c", "--count", type=int, help="How many images the model will be tested on.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Seed for dataset shuffling.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Make script verbose")
    args = parser.parse_args()

    GROUND_TRUTH, PREDICTIONS = EvalJob.yolo_val(
        Path(args.model_path),
        Path(args.images_path),
        Path(args.annot_path),
        args.count,
        seed=int(args.seed),
        verbose=args.verbose,
    )

    GROUND_TRUTH: list[ValBoundingBox]
    PREDICTIONS: list[ValBoundingBox]
    global_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    scores = FScores.collect_f_scores(
        GROUND_TRUTH,
        PREDICTIONS,
        CLASSES,
        iou_thresholds=global_thresholds,
        verbose=args.verbose,
    )

    FScores.plot_f_scores(global_thresholds, scores, CLASSES + ["all"])

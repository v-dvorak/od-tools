import csv
import io
import json
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from prettytable import PrettyTable, TableStyle
from tqdm import tqdm

from ..annotations import Annotation, AnnotationType, FullPage
from ..conversions.formats import Format

SUMMARY_FILE_NAME = "summary"
SCALE_FILE_PREFIX = "scale"

# COCO's default cap of 100 detections per image per class is too low
# for dense music pages (600+ noteheads on a single page)
MAX_DETECTIONS = 1000

# stats_as_dict keys reported in the summary,
# mapped to their human-readable names
SUMMARY_METRICS = {
    "AP_all": "AP50:95",
    "AP_50": "AP50",
    "AP_75": "AP75",
    f"AR_{MAX_DETECTIONS}": f"AR{MAX_DETECTIONS}",
}

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "tif", "tiff", "bmp")


def run_coco_eval(
    pred_dir: Path,
    label_dir: Path,
    class_reference_table: dict[str, int],
    class_output_names: list[str],
    pred_format: Format = Format.MUNG,
    label_format: Format = Format.MUNG,
    output_dir: Path | None = None,
    image_format: str = "jpg",
    rescale: bool = False,
    scale_start: float = 0.9,
    scale_end: float = 1.1,
    scale_step: float = 0.05,
    scale_gt: bool = False,
    confidence_threshold: float = 0.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Loads predictions and labels, matches them by name and evaluates
    the predictions with COCO metrics.

    If rescale is set, annotations are additionally evaluated rescaled
    by every scale in the range [scale_start, scale_end] with the given
    step. Scaling keeps each annotation's center and scales its size.
    Scale 1.0 (no rescaling) is always part of the sweep.
    The best scale is decided by AP50:95 of all classes combined.

    By default the predictions are rescaled, if scale_gt is set,
    the ground truth is rescaled instead.

    Predictions with confidence below confidence_threshold
    are left out of the evaluation.

    :param pred_dir: directory with predicted annotations
    :param label_dir: directory with label (ground truth) annotations
    :param class_reference_table: class reference table
    :param class_output_names: class output names
    :param pred_format: format of predicted annotations
    :param label_format: format of label annotations
    :param output_dir: where results are saved, defaults to eval/<pred_dir name>
    :param image_format: expected image format
    :param rescale: whether to try rescaling
    :param scale_start: first scale tried when rescaling
    :param scale_end: last scale tried when rescaling
    :param scale_step: step between tried scales when rescaling
    :param scale_gt: whether to rescale the ground truth instead of the predictions
    :param confidence_threshold: predictions below this confidence are not evaluated
    :param verbose: make script verbose
    :return: summary dictionary, also saved as summary.json
    """
    pred_dir = Path(pred_dir)
    label_dir = Path(label_dir)
    if output_dir is None:
        # resolve, so that "." and ".." still produce a directory name
        output_dir = Path("eval") / pred_dir.resolve().name
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = _load_pages(
        pred_dir,
        label_dir,
        pred_format,
        label_format,
        class_reference_table,
        class_output_names,
        image_format=image_format,
    )

    if rescale:
        scales = _generate_scales(scale_start, scale_end, scale_step)
    else:
        scales = [1.0]

    # when scaling the ground truth, it has to be rebuilt for every scale
    gt_data = (
        None
        if scale_gt
        else _build_ground_truth(pages, class_reference_table, class_output_names)
    )
    counts = _class_counts(pages, confidence_threshold)

    results: dict[float, dict[str, float]] = {}
    per_class_results: dict[float, list[dict[str, Any]]] = {}
    for scale in tqdm(scales, desc="Evaluating scales"):
        if scale_gt:
            gt_data = _build_ground_truth(
                pages, class_reference_table, class_output_names, scale
            )
            predictions = _build_predictions(pages, 1.0, confidence_threshold)
        else:
            predictions = _build_predictions(pages, scale, confidence_threshold)
        if not predictions:
            print(f"Warning: no predictions with confidence >= {confidence_threshold}")
        stats, per_class = _run_evaluation(gt_data, predictions, verbose=verbose)  # type: ignore
        results[scale] = stats
        per_class = _merge_class_counts(per_class, counts)
        per_class_results[scale] = per_class

        with open(
            output_dir / f"{SCALE_FILE_PREFIX}-{scale:.3f}.json", "w"
        ) as output_file:
            json.dump(
                {
                    "scale": scale,
                    "stats": stats,
                    "per_class": per_class,
                },
                output_file,
                indent=4,
            )

    best_scale = max(results, key=lambda s: results[s]["AP_all"])

    summary = {
        "pred_dir": str(pred_dir),
        "label_dir": str(label_dir),
        "pred_format": str(pred_format),
        "label_format": str(label_format),
        "image_count": len(pages),
        "confidence_threshold": confidence_threshold,
        "scale_gt": scale_gt,
        "best_scale": best_scale,
        "scales": [
            {
                "scale": scale,
                **{name: results[scale][key] for key, name in SUMMARY_METRICS.items()},
                "per_class": per_class_results[scale],
            }
            for scale in scales
        ],
    }
    with open(output_dir / f"{SUMMARY_FILE_NAME}.json", "w") as output_file:
        json.dump(summary, output_file, indent=4)
    with open(output_dir / f"{SUMMARY_FILE_NAME}.csv", "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["scale"] + list(SUMMARY_METRICS.values()))
        for scale in scales:
            writer.writerow(
                [f"{scale:.3f}"]
                + [f"{results[scale][key]:.4f}" for key in SUMMARY_METRICS]
            )

    table = PrettyTable(["scale"] + list(SUMMARY_METRICS.values()))
    for scale in scales:
        table.add_row(
            [f"{scale:.3f}"] + [f"{results[scale][key]:.4f}" for key in SUMMARY_METRICS]
        )
    if verbose:
        table.set_style(TableStyle.MARKDOWN)
    print(table)
    print(
        f"Best scale: {best_scale:.3f} ({SUMMARY_METRICS['AP_all']} = {results[best_scale]['AP_all']:.4f})"
    )

    return summary


def _load_pages(
    pred_dir: Path,
    label_dir: Path,
    pred_format: Format,
    label_format: Format,
    class_reference_table: dict[str, int],
    class_output_names: list[str],
    image_format: str = "jpg",
) -> list[tuple[str, FullPage, FullPage]]:
    """
    Loads prediction and label pages for every prediction file
    in the prediction directory.

    Labels are matched to predictions by file stem and are searched for
    in the label directory, predictions and their images in the
    prediction directory.

    :return: list of (stem, prediction page, label page) tuples
    """
    pred_extension = pred_format.to_annotation_extension()
    label_extension = label_format.to_annotation_extension()

    pred_files = sorted(pred_dir.glob(f"*.{pred_extension}"))
    if not pred_files:
        raise ValueError(f"No '{pred_extension}' prediction files found in {pred_dir}")

    pages = []
    for pred_file in tqdm(pred_files, desc="Loading pages"):
        stem = pred_file.stem

        label_file = _find_annotation_file(label_dir, stem, label_extension)
        if label_file is None:
            print(f"Warning: no label found for '{stem}', skipping")
            continue

        pred_image = _find_image_file(pred_dir, stem, image_format)
        if pred_image is None and pred_format in (Format.COCO, Format.DOLORES_COCO):
            pred_image = pred_file
        if pred_image is None:
            raise ValueError(f"No image found for prediction '{stem}'")

        label_image = _find_image_file(label_dir, stem, image_format)
        if label_image is None:
            label_image = pred_image

        pred_page = FullPage.load_from_file(
            pred_file,
            pred_image,
            class_reference_table,
            class_output_names,
            pred_format,
            an_type=AnnotationType.PREDICTION,
            with_confidence=True,
        )
        label_page = FullPage.load_from_file(
            label_file,
            label_image,
            class_reference_table,
            class_output_names,
            label_format,
            an_type=AnnotationType.GROUND_TRUTH,
        )

        pages.append((stem, pred_page, label_page))

    if not pages:
        raise ValueError(
            f"No loadable prediction/label pairs found ({pred_dir} vs {label_dir})"
        )
    return pages


def _find_annotation_file(directory: Path, stem: str, extension: str) -> Path | None:
    """
    Finds an annotation file for the given stem.

    Tries, in order:
      1. ``{directory}/{stem}.{extension}``
      2. ``{directory}/{stem}/*.{extension}`` (per-image subdirectory)
      3. ``{directory}/**/{stem}.{extension}`` (recursive fallback)

    :return: path to the annotation file or None if not found
    """
    candidate = directory / f"{stem}.{extension}"
    if candidate.is_file():
        return candidate

    subdir = directory / stem
    if subdir.is_dir():
        matches = sorted(subdir.glob(f"*.{extension}"))
        if matches:
            return matches[0]

    matches = sorted(directory.rglob(f"{stem}.{extension}"))
    return matches[0] if matches else None


def _find_image_file(directory: Path, stem: str, image_format: str) -> Path | None:
    """
    Finds an image for the given stem, same search order
    as in _find_annotation_file.

    :return: path to the image or None if not found
    """
    extensions = [image_format] + [
        extension for extension in IMAGE_EXTENSIONS if extension != image_format
    ]

    for extension in extensions:
        candidate = directory / f"{stem}.{extension}"
        if candidate.is_file():
            return candidate

    subdir = directory / stem
    if subdir.is_dir():
        for extension in extensions:
            matches = sorted(subdir.glob(f"*.{extension}"))
            if matches:
                return matches[0]

    for extension in extensions:
        matches = sorted(directory.rglob(f"{stem}.{extension}"))
        if matches:
            return matches[0]

    return None


def _generate_scales(start: float, end: float, step: float) -> list[float]:
    """
    Generates scales in the range [start, end] with the given step.
    Scale 1.0 is always part of the result.

    :return: sorted list of scales
    """
    if step <= 0:
        raise ValueError(f"Scale step must be positive, got {step}")
    if end < start:
        raise ValueError(
            f"Scale end must not be smaller than start, got {start} .. {end}"
        )

    count = round((end - start) / step) + 1
    scales = [round(start + i * step, 6) for i in range(count)]
    # drop overshooting last step caused by rounding
    scales = [scale for scale in scales if scale <= end + 1e-9]

    if 1.0 not in scales:
        scales.append(1.0)
    return sorted(set(scales))


def _scaled_bbox(annot: Annotation, scale: float) -> list[float]:
    """
    Returns the annotation's bbox as [left, top, width, height],
    shrunk or expanded around its center by the given scale.

    :return: scaled bbox as a list of floats
    """
    center_x = annot.bbox.left + annot.bbox.width / 2
    center_y = annot.bbox.top + annot.bbox.height / 2
    width = annot.bbox.width * scale
    height = annot.bbox.height * scale

    return [center_x - width / 2, center_y - height / 2, width, height]


def _build_ground_truth(
    pages: list[tuple[str, FullPage, FullPage]],
    class_reference_table: dict[str, int],
    class_output_names: list[str],
    scale: float = 1.0,
) -> dict[str, Any]:
    """
    Builds a COCO ground truth dataset from label pages, with every
    annotation shrunk or expanded around its center by the given scale.

    :return: COCO dataset dictionary
    """
    categories = [
        {"id": class_reference_table[name], "name": name} for name in class_output_names
    ]

    images_data = []
    annotations_data = []
    annot_id = 1
    for image_id, (stem, _, label_page) in enumerate(pages, start=1):
        images_data.append(
            {
                "id": image_id,
                "file_name": stem,
                "width": label_page.width,
                "height": label_page.height,
            }
        )
        for annot in label_page.all_annotations():
            left, top, width, height = _scaled_bbox(annot, scale)
            annotations_data.append(
                {
                    "id": annot_id,
                    "image_id": image_id,
                    "category_id": annot.class_id,
                    "bbox": [left, top, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annot_id += 1

    return {
        "images": images_data,
        "annotations": annotations_data,
        "categories": categories,
    }


def _build_predictions(
    pages: list[tuple[str, FullPage, FullPage]],
    scale: float = 1.0,
    confidence_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Builds a list of COCO predictions from prediction pages,
    with every annotation shrunk or expanded around its center
    by the given scale. Annotations with confidence below
    the threshold are skipped.

    :return: list of COCO prediction dictionaries
    """
    predictions = []
    for image_id, (_, pred_page, _) in enumerate(pages, start=1):
        for annot in pred_page.all_annotations():
            if annot.confidence < confidence_threshold:
                continue

            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": annot.class_id,
                    "bbox": _scaled_bbox(annot, scale),
                    "score": annot.confidence,
                }
            )
    return predictions


def _class_counts(
    pages: list[tuple[str, FullPage, FullPage]],
    confidence_threshold: float = 0.0,
) -> dict[str, dict[str, int]]:
    """
    Counts ground truth and predicted annotations per class name,
    with an "all" entry holding the totals.

    Predictions below the confidence threshold are not counted.

    :return: mapping of class name to {"gt_count": n, "pred_count": m}
    """
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"gt_count": 0, "pred_count": 0}
    )
    counts["all"] = {"gt_count": 0, "pred_count": 0}

    for _, pred_page, label_page in pages:
        for annot in label_page.all_annotations():
            counts[annot.class_name]["gt_count"] += 1
            counts["all"]["gt_count"] += 1
        for annot in pred_page.all_annotations():
            if annot.confidence >= confidence_threshold:
                counts[annot.class_name]["pred_count"] += 1
                counts["all"]["pred_count"] += 1

    return counts


def _merge_class_counts(
    per_class: list[dict[str, Any]],
    counts: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    """
    Adds ground truth and predicted counts to each per-class metric entry.

    Classes that have predictions but no ground truth have no metrics
    entry, for those a counts-only entry is appended.

    :return: list of per-class dictionaries with counts
    """
    merged = []
    for entry in per_class:
        name = entry["class"]
        merged.append(
            {
                "class": name,
                **counts.get(name, {"gt_count": 0, "pred_count": 0}),
                **{key: value for key, value in entry.items() if key != "class"},
            }
        )

    known = {entry["class"] for entry in per_class}
    for name, class_counts in counts.items():
        if name not in known:
            merged.append({"class": name, **class_counts})

    return merged


def _run_evaluation(
    gt_data: dict[str, Any],
    predictions: list[dict[str, Any]],
    verbose: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """
    Runs COCO evaluation of predictions against the ground truth.

    :return: tuple of COCO metrics and per-class metrics
    """
    from faster_coco_eval import COCO, COCOeval_faster

    coco_gt = COCO(gt_data)
    coco_pred = coco_gt.loadRes(predictions)

    coco_eval = COCOeval_faster(coco_gt, coco_pred, "bbox")
    # AP is computed from the last entry of maxDets
    coco_eval.params.maxDets = [1, 10, MAX_DETECTIONS]
    coco_eval.evaluate()
    coco_eval.accumulate()
    if verbose:
        coco_eval.summarize()
    else:
        # summarize computes the stats but also prints them
        with redirect_stdout(io.StringIO()):
            coco_eval.summarize()

    return coco_eval.stats_as_dict, coco_eval.extended_metrics["class_map"]

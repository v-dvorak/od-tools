from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from . import COCOEvaluator
from ..Conversions.Annotations import Annotation

font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")


def calculate_f1_score(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 0
    return 2 * tp / (2 * tp + fp + fn)


def plot_f_scores(iou_thresholds: list[float], f1_scores_per_class: list[list[float]], class_labels: list[str | int]):
    """
    Plots F1 score for different IoU thresholds for multiple classes.

    :param iou_thresholds: List of IoU thresholds.
    :param f1_scores_per_class: List of lists containing F1 score for each class.
    :param class_labels: List of class labels.
    """
    plt.figure(figsize=(10, 6))

    plt.xticks(np.arange(min(iou_thresholds), max(iou_thresholds) + 0.05, 0.05))

    # Plot F1 score for each class
    for i, f1_scores in enumerate(f1_scores_per_class):
        plt.plot(iou_thresholds, f1_scores, marker='o', label=class_labels[i])

    # Labels and title
    plt.title('F1 Scores for Different IoU Thresholds')
    plt.xlabel('IoU Thresholds')
    plt.ylabel('F1 Score')
    plt.xlim(min(iou_thresholds), 1)
    plt.ylim(0, 1.05)

    # Show legend for the classes
    plt.legend(title="Classes")

    # Show grid
    plt.grid(True)

    # Display plot
    plt.show()


def collect_f_scores(
        ground_truth: list[Annotation],
        predictions: list[Annotation],
        class_names: list[str],
        iou_thresholds: list[float] = None,
        summation: bool = True,
        verbose: bool = False,
):
    """
    Given ground truths and predictions, collect F1 scores for different IoU thresholds.

    :param ground_truth: list of ground truth bounding boxes
    :param predictions: list of predicted bounding boxes
    :param class_names: list of class labels
    :param iou_thresholds: list of IoU thresholds
    :param summation: add summary (over all classes) f1 score for each IoU threshold
    :param verbose: make script verbose

    :return: list of F1 scores for each IoU threshold
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    total_classes = len(class_names)

    if summation:
        class_names.append("all")

    collected_scores = [[] for _ in range(len(class_names))]
    for threshold in iou_thresholds:
        if verbose:
            print(f'IoU threshold: {threshold}')

        all_tp, all_fp, all_fn = 0, 0, 0
        res = COCOEvaluator.get_coco_metrics(ground_truth, predictions, max_detections=500, iou_threshold=threshold)

        # collect classes
        for class_id in range(total_classes):
            if class_id not in res.keys():
                if verbose:
                    print(f"{class_id}: F1 score 0")
                collected_scores[class_id].append(0)
            else:
                if verbose:
                    print(
                        f"{class_id}: F1 score {calculate_f1_score(res[class_id]['TP'], res[class_id]['FP'], res[class_id]['FN'])}")

                all_tp += res[class_id]['TP']
                all_fp += res[class_id]['FP']
                all_fn += res[class_id]['FN']
                collected_scores[class_id].append(
                    calculate_f1_score(res[class_id]['TP'], res[class_id]['FP'], res[class_id]['FN']))

        if summation:
            if verbose:
                print(f"All: F1 score {calculate_f1_score(all_tp, all_fp, all_fn)}")

            collected_scores[-1].append(calculate_f1_score(all_tp, all_fp, all_fn))

    return collected_scores

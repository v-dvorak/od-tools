from collections import defaultdict

import numpy as np

from ..Conversions.Annotations.Annotation import Annotation


def _group_detections(dt: list[Annotation], gt: list[Annotation]):
    """
    Group ground truths and detections on a image x class basis.
    """
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for d in dt:
        i_id = d.get_image_name()
        bb_info[i_id]["dt"].append(d)
    for g in gt:
        i_id = g.get_image_name()
        bb_info[i_id]["gt"].append(g)
    return bb_info


def compute_class_confusion_matrix(ground_truths: list[Annotation], predictions: list[Annotation], num_classes: int,
                                   iou_threshold: float = 0.9) -> dict[str, np.ndarray]:
    """
    Computes a confusion matrix for each class considering mislabeling between classes,
    as well as false positives and false negatives with respect to background.

    :param ground_truths: List of ground truth annotations.
    :param predictions: List of predicted annotations.
    :param num_classes: Total number of classes (excluding background).
    :param iou_threshold: Intersection over Union pixel_threshold for a valid match.
    :return: Confusion matrix dictionary with both absolute and relative values.
    """
    # Create a (num_classes + 1) x (num_classes + 1) confusion matrix
    # Last index for background confusion
    bb_info = _group_detections(ground_truths, predictions)

    confusion_matrix_abs = np.zeros((num_classes + 1, num_classes + 1), dtype=int)

    # Track matched annotations to avoid double-counting

    # Loop over predictions and ground truths to find matches based on IoU
    # TODO: process by image
    for img_id in bb_info.keys():
        ground_truths, predictions = bb_info[img_id]["gt"], bb_info[img_id]["dt"]
        matched_ground_truths = set()
        matched_predictions = set()
        for pred in predictions:
            pred_class = pred.get_class_id()
            matched = False

            for gt in ground_truths:
                gt_class = gt.get_class_id()

                # Only consider annotations that intersect and meet IoU pixel_threshold
                if gt_class == pred_class and gt.intersects(pred):
                    iou = gt.bbox.intersection_over_union(pred.bbox)

                    if iou >= iou_threshold:
                        confusion_matrix_abs[gt_class, pred_class] += 1  # True Positive for correct match
                        matched_ground_truths.add(id(gt))
                        matched_predictions.add(id(pred))
                        matched = True
                        break

            if not matched:
                # Prediction that did not match any ground truth; it's a False Positive
                confusion_matrix_abs[num_classes, pred_class] += 1

        # Now, calculate False Negatives (missed detections) for ground truths that were not matched
        for gt in ground_truths:
            gt_class = gt.get_class_id()
            if id(gt) not in matched_ground_truths:
                confusion_matrix_abs[gt_class, num_classes] += 1  # Missed detection, count as FN for that class

    # Convert absolute values to relative values by dividing by total detections (for each row)
    total_detections_per_class = confusion_matrix_abs.sum(axis=1, keepdims=True)
    confusion_matrix_rel = np.divide(
        confusion_matrix_abs,
        total_detections_per_class,
        out=np.zeros_like(confusion_matrix_abs, dtype=float),
        where=total_detections_per_class != 0
    )
    # confusion_matrix_rel = confusion_matrix_abs / total_detections_per_class.sum()

    return {
        'absolute': confusion_matrix_abs,
        'relative': confusion_matrix_rel
    }


# Example usage
# Assume ground_truths and predictions are lists of `Annotation` objects
# and num_classes is the number of object classes (e.g., for "car", "cat", etc.)
# num_classes = 5  # Example number of classes

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(matrix: np.ndarray, class_labels: list, title: str = "Confusion Matrix"):
    """
    Plots a confusion matrix using seaborn's heatmap.

    :param matrix: The confusion matrix to plot.
    :param class_labels: List of class labels, including 'Background' for the last class.
    :param title: Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(matrix, cmap="Blues", cbar=False, square=True, vmin=1, vmax=np.max(matrix),
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)

    # Add titles and axis labels
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(title)
    plt.show()

# Example usage
# Assuming `confusion_matrix['absolute']` contains the computed confusion matrix
# and `num_classes` is defined

# # Add 'Background' as the last class label for plotting
# class_labels = [f"Class {i}" for i in range(num_classes)] + ["Background"]
#
# # Plot the absolute confusion matrix
# plot_confusion_matrix(confusion_matrix['absolute'], class_labels, title="Absolute Confusion Matrix")
#
# # Plot the relative confusion matrix
# plot_confusion_matrix((confusion_matrix['relative'] * 100).astype(int), class_labels,
#                       title="Relative Confusion Matrix (%)")

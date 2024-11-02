import random
from enum import Enum
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from . import Utils
from ..Conversions import ConversionUtils
from ..Conversions.Annotations import FullPage, Annotation
from ..Conversions.Formats import InputFormat
from ..Splitting import SplitUtils
from ..Val import FScores


class ModelType(Enum):
    YOLO_DETECTION = 1
    YOLO_SEGMENTATION = 2

    @staticmethod
    def from_string(model_type: str):
        """
        Given string, returns a model type.

        :param model_type: string definition of model
        :return: enum model type
        """
        model_type = model_type.lower()
        if model_type == "yolod":
            return ModelType.YOLO_DETECTION
        elif model_type == "yolos":
            return ModelType.YOLO_SEGMENTATION
        else:
            raise ValueError(f"Invalid model type: {model_type}")


GLOBAL_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]


def run_f1_scores_vs_iou(
        model_path: Path,
        images_path: Path,
        annotations_path: Path,
        input_format: InputFormat,
        model_type: ModelType,
        class_output_names: list[str],

        output_dir: Path = None,
        summation: bool = True,
        count: int = None,
        verbose: bool = False,
):
    """
    Load ground truths, retrieve predictions and compute f1 scores.

    :param model_path: Path to the model weights
    :param images_path: Path to the image folder
    :param annotations_path: Path to the annotation folder
    :param input_format: annotation input format
    :param model_type: model type
    :param class_output_names: list of class names
    :param output_dir: path to output directory
    :param summation: whether to add "All" category to f1 scores
    :param count: number of images to process
    :param verbose: make script verbose
    """
    # this is here just to force this into loader methods
    # without changing anything about / merging the loaded classes
    class_reference_table = {}
    for i, class_name in enumerate(class_output_names):
        class_reference_table[class_name] = i

    ground_truths, predictions = get_gts_and_predictions(
        model_path,
        images_path,
        annotations_path,
        input_format,
        model_type,
        class_output_names,
        class_reference_table,
        count=count,
        verbose=verbose,
        debug=False,
    )

    scores = FScores.collect_f_scores(
        ground_truths,
        predictions,
        class_output_names,
        iou_thresholds=GLOBAL_IOU_THRESHOLDS,
        verbose=verbose,
        summation=summation,
    )

    FScores.plot_f_scores(
        GLOBAL_IOU_THRESHOLDS,
        scores,
        class_output_names + ["all"] if summation else class_output_names,
        output_path=output_dir / "f1-scores.png" if output_dir is not None else None,
    )


def retrieve_ground_truth(
        image_path: Path,
        annotation_path: Path,
        input_format: InputFormat,
        class_output_names: list[str],
        class_reference_table: dict[str, int],
        index: int,
        debug: bool = False,
) -> list[Annotation]:
    """
    Load ground truth from single annotation file.

    :param image_path: Path to the image
    :param annotation_path: Path to the annotation file
    :param input_format: annotation input format
    :param class_output_names: list of class names
    :param class_reference_table: class reference table
    :param index: index of the image
    :param debug: show loaded annotations as rectangles on image
    :return: list of ground truths
    """
    ground_truth = FullPage.load_from_file(
        annotation_path,
        image_path,
        class_reference_table,
        class_output_names,
        input_format
    )

    if debug:
        SplitUtils.draw_rectangles_on_image(
            image_path.__str__(),
            [annot.bbox for annot in ground_truth.annotations()],
            color=(0, 255, 0),
            thickness=2,
            # output_path=f"pred1-{index}.jpg"
        )
        input("..")

    # force image id upon gts
    for annot in ground_truth.all_annotations():
        annot.set_image_name(str(index))

    return list(ground_truth.all_annotations())


def predict_yolo_split(
        model,
        image_path: Path,
        index: int,
        overlap: float = 0.25,
        debug: bool = False,
) -> list[Annotation]:
    """
    Split image into tiles, predict for each of them, resolve and return predictions.

    :param model: loaded model
    :param image_path: Path to the image
    :param index: index of the image
    :param overlap: overlap ratio
    :param debug: show loaded annotations as rectangles on image
    :return: list of predictions
    """
    width, height = ConversionUtils.get_num_pixels(image_path)

    # prepare images for inference
    splits = SplitUtils.create_split_box_matrix((width, height), overlap_ratio=overlap)
    tiles = Utils.create_split_images(image_path, splits)
    # predict
    results = model.predict(tiles)

    # collect data from subpages
    subpages = []
    for i in range(len(results)):
        result = results[i]
        res = FullPage.from_yolo_result(result)
        subpages.append(res)

    resolved = FullPage.combine_multiple_pages_and_resolve(subpages, splits)

    if debug:
        SplitUtils.draw_rectangles_on_image(
            image_path.__str__(),
            [annot.bbox for annot in resolved.all_annotations()],
            color=(0, 255, 0),
            thickness=2,
            # output_path=f"pred1-{index}.jpg"
        )
        input("..")

    for annot in resolved.all_annotations():
        annot.set_image_name(str(index))

    return list(resolved.all_annotations())


def get_gts_and_predictions(
        model_path: Path,
        images_path: Path,
        annotations_path: Path,
        input_format: InputFormat,
        model_type: ModelType,
        class_output_names: list[str],
        class_reference_table: dict[str, int],
        image_format: str = "jpg",
        count: int = None,
        verbose: bool = False,
        seed: int = 42,
        overlap: float = 0.25,
        # image_splitting: bool = False,
        debug: bool = False,
) -> tuple[list[Annotation], list[Annotation]]:
    """
    Loads ground truths and predictions from validation dataset using the given model.

    :param model_path: Path to the model weights
    :param images_path: Path to the image folder
    :param annotations_path: Path to the annotation folder
    :param input_format: annotation input format
    :param model_type: model type
    :param class_output_names: list of class names
    :param class_reference_table: class reference table
    :param image_format: image format
    :param count: number of images to process
    :param verbose: show loaded annotations as rectangles on image
    :param seed: seed for reproducibility
    :param overlap: overlap ratio
    :param debug: show loaded annotations as rectangles on image
    :return: list of ground truths and list of predictions
    """
    # TODO implement evaluation without image splitting
    # load validation data
    images = sorted(list(images_path.rglob(f"*.{image_format}")))
    annotations = sorted(list(annotations_path.rglob(f"*.{input_format.to_annotation_extension()}")))
    assert len(images) == len(annotations)
    data = list(zip(images, annotations))

    if verbose:
        print(f"Loaded {len(images)} images:")
        for im in images:
            print(im.stem)
        print()

    # load model
    if model_type == ModelType.YOLO_DETECTION or model_type == ModelType.YOLO_SEGMENTATION:
        model = YOLO(model_path)
    else:
        raise NotImplementedError(f'Not implemented model type: {model_type}')

    # set up optional arguments
    if count is not None:
        random.Random(seed).shuffle(images)
        data = data[:count]

    ground_truths: list[Annotation] = []
    predictions: list[Annotation] = []

    # predict for every single image
    index = 0
    for image, annotation in tqdm(data, desc="Validating model"):
        ground_truths += retrieve_ground_truth(
            image,
            annotation,
            input_format,
            class_output_names,
            class_reference_table,
            index,
            debug=debug,
        )
        predictions += predict_yolo_split(
            model,
            image,
            index,
            overlap=overlap,
            debug=debug
        )
        index += 1

    return ground_truths, predictions

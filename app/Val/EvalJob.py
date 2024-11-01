import random
from enum import Enum
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from odmetrics.bounding_box import ValBoundingBox
from . import Utils
from ..Conversions import ConversionUtils
from ..Conversions.BoundingBox import BoundingBox
from ..Conversions.COCO.AnnotationClasses import COCOFullPage
from ..Conversions.Formats import InputFormat
from ..Splitting import SplitUtils


class ModelType(Enum):
    YOLO_DETECTION = 1
    YOLO_SEGMENTATION = 2


def retrieve_ground_truth(
        image_path: Path,
        annotation_path: Path,
        input_format: InputFormat,
        class_output_names: list[str],
        class_reference_table: dict[str, int],
        index: int,
        debug: bool = False,

) -> list[ValBoundingBox]:
    ground_truths: list[ValBoundingBox] = []

    gts = COCOFullPage.load_from_file(
        annotation_path,
        image_path,
        class_reference_table,
        class_output_names,
        input_format
    )

    for gt in gts.all_annotations():
        ground_truths.append(gt.to_val_box(str(index), ground_truth=True))

    if debug:
        SplitUtils.draw_rectangles_on_image(
            image_path.__str__(),
            [BoundingBox(int(box._x), int(box._y), int(box._x2), int(box._y2)) for box in ground_truths],
            color=(0, 255, 0),
            thickness=2,
            # output_path=f"pred1-{index}.jpg"
        )
        input("..")
    return ground_truths


def predict_yolo_split(
        model,
        image_path: Path,
        index: int,
        overlap: float = 0.25,
        debug: bool = False,
) -> list[ValBoundingBox]:
    predictions: list[ValBoundingBox] = []
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
        res = COCOFullPage.from_yolo_result(result)
        subpages.append(res)

    resolved = COCOFullPage.combine_multiple_pages_and_resolve(subpages, splits)

    for pred in resolved.all_annotations():
        predictions.append(pred.to_val_box(str(index), ground_truth=False))

    if debug:
        SplitUtils.draw_rectangles_on_image(
            image_path.__str__(),
            [BoundingBox(int(box._x), int(box._y), int(box._x2), int(box._y2)) for box in predictions],
            color=(0, 255, 0),
            thickness=2,
            # output_path=f"pred1-{index}.jpg"
        )
        input("..")

    return predictions


def validate_model(
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
        image_splitting: bool = False,
        debug: bool = False,
):
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

    ground_truths: list[ValBoundingBox] = []
    predictions: list[ValBoundingBox] = []

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

import random
from enum import Enum
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from . import Utils
from ..Conversions import ConversionUtils
from ..Conversions.BoundingBox import BoundingBox
from ..Conversions.COCO.AnnotationClasses import COCOFullPage, COCOAnnotation
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

) -> list[COCOAnnotation]:

    ground_truth = COCOFullPage.load_from_file(
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
) -> list[COCOAnnotation]:
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
) -> tuple[list[COCOAnnotation], list[COCOAnnotation]]:
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

    ground_truths: list[COCOAnnotation] = []
    predictions: list[COCOAnnotation] = []

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

import random
from pathlib import Path

from tqdm import tqdm

from app.Conversions.BoundingBox import BoundingBox
from app.Conversions.COCO.AnnotationClasses import COCOFullPage
from app.Splitting import SplitUtils
from odmetrics.bounding_box import ValBoundingBox
from . import Utils


def yolo_val(
        model_path: Path,
        images_path: Path,
        annotations_path: Path,
        count: int = None,
        overlap: float = 0.25,
        seed: int = 42,
        image_format: str = "jpg",
        annotation_format: str = "txt",
        verbose: bool = False,
) -> tuple[list[ValBoundingBox], list[ValBoundingBox]]:
    model = Utils.load_model(model_path.__str__())
    images = sorted(list(images_path.rglob(f"*.{image_format}")))
    annotations = sorted(list(annotations_path.rglob(f"*.{annotation_format}")))
    assert len(images) == len(annotations)
    data = list(zip(images, annotations))

    if verbose:
        print(f"Loaded {len(images)} images:")
        for im in images:
            print(im.stem)
        print()

    if count is not None:
        random.Random(seed).shuffle(images)
        data = data[:count]

    ground_truths = []
    predictions = []

    index = 0
    for image, annotation in tqdm(data):
        _, (width, height) = Utils.prepare_image(image)

        splits = SplitUtils.create_split_box_matrix((width, height), overlap_ratio=overlap)
        tiles = Utils.create_split_images(image, splits)
        results = model.predict(tiles)

        subpages = []
        for i in range(len(results)):
            result = results[i]
            res = COCOFullPage.from_yolo_result(result)
            subpages.append(res)

        resolved = COCOFullPage.combine_multiple_pages_and_resolve(subpages, splits)

        debug = False

        if debug:
            SplitUtils.draw_rectangles_on_image(
                image,
                [x.bbox for xs in resolved.annotations for x in xs],
                color=(0, 255, 0),
                thickness=2,
                output_path=f"pred-{index}.jpg"
            )

        # eval_utils.draw_rectangles(image, yolo_utils.prepare_prediction(prediction.boxes), 2)
        for pred in resolved.all_annotations():
            # predictions.append(coco_annot_to_bbx(pred, str(index), ground_truth=False))
            predictions.append(pred.to_val_box(str(index), ground_truth=False))
        # predictions.append(resolved.to_eval_format())
        from app.Conversions.YOLO.AnnotationClasses import YOLOFullPageDetection
        gts = YOLOFullPageDetection.from_yolo_file(annotation, (width, height))
        for gt in gts.annotations:
            ground_truths.append(gt.to_val_box(str(index), (width, height), ground_truth=True))

        if debug:
            SplitUtils.draw_rectangles_on_image(
                image,
                [BoundingBox(int(box._x), int(box._y), int(box._x2), int(box._y2)) for box in predictions],
                color=(0, 255, 0),
                thickness=2,
                output_path=f"pred1-{index}.jpg"
            )

        ground_truths: list[ValBoundingBox]

        if debug:
            SplitUtils.draw_rectangles_on_image(
                image,
                [BoundingBox(int(box._x), int(box._y), int(box._x2), int(box._y2)) for box in ground_truths],
                color=(0, 255, 0),
                thickness=2,
                output_path=f"pred1-{index}.jpg"
            )

        if debug:
            print(predictions)
            print(ground_truths)

        index += 1

    return ground_truths, predictions

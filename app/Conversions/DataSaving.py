import json
from pathlib import Path

import cv2

from .COCO.AnnotationClasses import COCOFullPageEncoder
from .COCO.AnnotationClasses import COCOSplitPage
from .COCO.Interfaces import ICOCOFullPage
from .YOLO.AnnotationClasses import YOLOFullPageDetection, YOLOFullPageSegmentation
from ..Splitting.SplitUtils import BoundingBox


def _save_split_page_image(
        path_to_image: Path,
        output_dir: Path,
        splits: list[list[BoundingBox]],
        image_format: str = "jpg",
) -> None:
    """
    Loads and splits image given by path and saves it to output_dir.
    Images are stored in format `"name"-"row"-"column"."image_format"`.

    :param path_to_image: image to process
    :param output_dir: directory to save all split images to
    :param splits: list of bounding boxes that define splits to save
    :param image_format: format of image to save to
    """
    img = cv2.imread(path_to_image.__str__())
    dato_name = path_to_image.stem

    for row in range(len(splits)):
        for col in range(len(splits[0])):
            # save subpage images

            cutout = splits[row][col]
            sub_img = img[cutout.top:cutout.bottom, cutout.left:cutout.right]

            # DEBUG viz
            # SplitUtils.draw_rectangles_on_image(
            #     sub_img,
            #     [SplitUtils.BoundingBox.from_coco_annotation(a) for a in full_page.subpages[row][col].annotations[0]],
            #     thickness=2,
            #     loaded=True,
            #     color=(255, 0, 0),
            # )

            # input("..")
            cv2.imwrite((output_dir / (dato_name + f"-{row}-{col}.{image_format}")).__str__(), sub_img)


def _save_split_page_coco_annotation(
        dato_name: str,
        output_dir: Path,
        split_page: COCOSplitPage
) -> None:
    """
    Processes already split COCO annotations (stored in `COCOSplitPage`) and saves them to output_dir in COCO format.
    Annotations are stored in format `"dato_name"-"row"-"column".json`.

    :param dato_name: name of the piece of data, defines names of output files
    :param output_dir: directory to save all split images to
    :param split_page: COCOSplitPage instance
    """
    # save subpage annotations
    for row in range(len(split_page.subpages)):
        for col in range(len(split_page.subpages[0])):
            _save_full_page_coco_annotation(
                dato_name + f"-{row}-{col}",
                output_dir,
                split_page.subpages[row][col]
            )


def _save_full_page_coco_annotation(
        dato_name: str,
        output_dir: Path,
        full_page: ICOCOFullPage
) -> None:
    """
    Processes COCO full page annotations and saves them to output_dir in COCO format.
    Annotations are stored in format `"dato_name"-"row"-"column".json`.

    :param dato_name: name of the piece of data, defines names of output files
    :param output_dir: directory to save all split images to
    :param full_page: COCOFullPage instance
    """
    with open(output_dir / (dato_name + ".json"), "w") as f:
        json.dump(full_page, f, indent=4, cls=COCOFullPageEncoder)


def _save_split_page_yolo_annotation_from_coco_split_page(
        dato_name: str,
        output_dir: Path,
        split_page: COCOSplitPage,
        mode: str = "detection",
) -> None:
    """
    Processes already split COCO annotations (stored in `COCOSplitPage`) and saves them to output_dir in YOLO format.
    Annotations are stored in format `"dato_name"-"row"-"column".txt`.

    :param dato_name: name of the piece of data, defines names of output files
    :param output_dir: directory to save all split images to
    :param split_page: COCOSplitPage instance
    :param mode: `detection` or `segmentation`
    """
    for row in range(len(split_page.subpages)):
        for col in range(len(split_page.subpages[0])):
            _save_full_page_yolo_annotation_from_coco_full_page(
                dato_name + f"-{row}-{col}",
                output_dir,
                split_page.subpages[row][col],
                mode=mode
            )


def _save_full_page_yolo_annotation_from_coco_full_page(
        dato_name: str,
        output_dir: Path,
        full_page: ICOCOFullPage,
        mode: str = "detection",
) -> None:
    """
    Processes COCO annotations (stored in `COCOFullPage`) and saves them to output_dir in YOLO format.
    Annotations are stored in format `"dato_name"-"row"-"column".txt`.

    :param dato_name: name of the piece of data, defines names of output files
    :param output_dir: directory to save all split images to
    :param full_page: COCOFullPage instance
    :param mode: `detection` or `segmentation`
    """
    if mode == "detection":
        yolo_fp = YOLOFullPageDetection.from_coco_page(full_page)
    elif mode == "segmentation":
        yolo_fp = YOLOFullPageSegmentation.from_coco_page(full_page)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    with open(output_dir / f"{dato_name}.txt", "w") as f:
        for annotation in yolo_fp.annotations:
            f.write(annotation.__str__() + "\n")

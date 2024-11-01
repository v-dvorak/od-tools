from __future__ import annotations

import os

import cv2
from PIL import Image
from ultralytics import YOLO

from app.Conversions.BoundingBox import BoundingBox

font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")


def create_split_images(image_path, splits: list[list[BoundingBox]]):
    output = []
    img = cv2.imread(image_path)
    for split in [x for xs in splits for x in xs]:
        # cv2.imshow("", img[split.top:split.bottom, split.left:split.right])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        output.append(img[split.top:split.bottom, split.left:split.right])
    return output


def load_model(model_path: str):
    return YOLO(model_path)


def prepare_image(image_path: str):
    image = Image.open(image_path)
    return image, image.size

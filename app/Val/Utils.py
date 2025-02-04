from __future__ import annotations

import os
from pathlib import Path

import cv2
from PIL import Image
from ultralytics import YOLO

font_path = os.path.join(cv2.__path__[0], "qt", "fonts", "DejaVuSans.ttf")


def load_model(model_path: str | Path):
    return YOLO(model_path)


def prepare_image(image_path: str | Path):
    image = Image.open(image_path)
    return image, image.size

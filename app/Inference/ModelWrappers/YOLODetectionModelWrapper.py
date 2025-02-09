import numpy as np
from ultralytics import YOLO
from pathlib import Path

from .ModelWrapper import IModelWrapper
from ...Conversions.Annotations.FullPage import FullPage


class YOLODetectionModelWrapper(IModelWrapper):
    """
    Implementation of YOLO detection model wrapper.
    """

    def __init__(self, model: YOLO | Path):
        if isinstance(model, Path) or isinstance(model, str):
            self.model = YOLO(model)
        else:
            self.model = model

    def predict_multiple(
            self,
            tiles: list[np.ndarray],
            wanted_ids: list[int] = None,
            verbose: bool = False
    ) -> list[FullPage]:
        predictions = self.model.predict(tiles, save=False, save_txt=False, verbose=verbose)
        subpages = []
        for prediction in predictions:
            subpages.append(FullPage.from_yolo_result(prediction, wanted_ids=wanted_ids))
        return subpages

    def predict_single(
            self,
            image: np.ndarray,
            wanted_ids: list[int] = None,
            verbose: bool = False
    ) -> FullPage:
        prediction = self.model.predict(image, save=False, save_txt=False, verbose=verbose)
        return FullPage.from_yolo_result(prediction[0], wanted_ids=wanted_ids)

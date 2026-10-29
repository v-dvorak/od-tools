import numpy as np
from ultralytics import YOLO
from pathlib import Path

from .model_wrapper import IModelWrapper
from ...annotations.full_page import FullPage


class YOLODetectionModelWrapper(IModelWrapper):
    """
    Implementation of YOLO detection model wrapper.
    """

    _WRAPPER_NAME = "YOLO detection"

    def __init__(self, model: YOLO | Path | str):
        if isinstance(model, (Path, str)):
            self.model = YOLO(model)
        else:
            self.model = model

    def predict_multiple(
        self,
        tiles: list[np.ndarray],
        wanted_ids: list[int] | None = None,
        verbose: bool = False,
        batch_size: int = 16,
    ) -> list[FullPage]:
        print(
            f"Running {self._WRAPPER_NAME} model for {len(tiles)} tiles with batch size {batch_size}"
        )
        predictions = self.model.predict(
            tiles, save=False, save_txt=False, verbose=verbose, batch=batch_size
        )
        subpages = []
        for prediction in predictions:
            subpages.append(
                FullPage.from_yolo_result(prediction, wanted_ids=wanted_ids)
            )
        return subpages

    def predict_single(
        self,
        image: np.ndarray,
        wanted_ids: list[int] | None = None,
        verbose: bool = False,
    ) -> FullPage:
        prediction = self.model.predict(
            image, save=False, save_txt=False, verbose=verbose
        )
        return FullPage.from_yolo_result(prediction[0], wanted_ids=wanted_ids)

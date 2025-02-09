import numpy as np

from .. import InferenceJob
from ...Conversions.Annotations.FullPage import FullPage


def yolo_multiple_detection(job: InferenceJob, tiles: list[np.ndarray], verbose: bool = False) -> list[FullPage]:
    predictions = job.model.predict(tiles, save=False, save_txt=False, verbose=verbose)
    subpages = []
    for prediction in predictions:
        subpages.append(FullPage.from_yolo_result(prediction))
    return subpages


def yolo_single_detection(job: InferenceJob, verbose: bool = False) -> FullPage:
    prediction = job.model.predict(job.image, save=False, save_txt=False, verbose=verbose)
    return FullPage.from_yolo_result(prediction[0], wanted_ids=job.wanted_ids)

import numpy as np

from .InferenceJob import InferenceJob
from .. import Splitting
from ..Conversions.Annotations.FullPage import FullPage


def _run_split_prediction_job(job: InferenceJob, verbose: bool = False) -> FullPage:
    # get image dimensions
    if isinstance(job.image, np.ndarray):
        w, h = job.image.shape[:2]
    else:
        w, h = job.image.size

    job.split_settings.update_window_size_based_on_tals(w if w > h else h)

    # create splits
    splits = Splitting.create_split_box_matrix(
        (h, w),
        window_size=(job.split_settings.width, job.split_settings.height),
        overlap_ratio=job.split_settings.overlap_ratio
    )
    tiles = Splitting.create_split_images(job.image, splits)

    # predict
    subpages: list[FullPage] = job.model.predict_multiple(tiles, wanted_ids=job.wanted_ids, verbose=verbose)

    # resolve
    resolved = FullPage.combine_multiple_pages_and_resolve(
        subpages,
        splits,
        iou_threshold=job.split_settings.iou_threshold,
        edge_offset=job.split_settings.edge_offset,
        verbose=verbose
    )

    return resolved


def _run_normal_prediction_job(job: InferenceJob, verbose: bool = False) -> FullPage:
    return job.model.predict_single(job.image, wanted_ids=job.wanted_ids, verbose=verbose)


def run_prediction_job(job: InferenceJob, verbose: bool = False) -> FullPage:
    """
    Runs given inference job.

    :param job: inference job
    :param verbose: make script verbose
    :return: predicted FullPage
    """
    if job.split_settings is None:
        return _run_normal_prediction_job(job, verbose=verbose)
    else:
        return _run_split_prediction_job(job, verbose=verbose)


def run_multiple_prediction_jobs(jobs: list[InferenceJob], verbose: bool = False) -> FullPage:
    """
    Runs given inference jobs and combines all outputs into a single FullPage.

    :param jobs: inference jobs
    :param verbose: make script verbose
    :return: predicted FullPage
    """
    fp: FullPage = None
    for job in jobs:
        result = run_prediction_job(job, verbose=verbose)
        if fp is None:
            fp = result
        else:
            fp.extend_page(result)

    return fp

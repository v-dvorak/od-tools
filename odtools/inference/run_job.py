from dataclasses import replace
from collections.abc import Iterable

import numpy as np

from .inference_job import InferenceJob
from .. import splitting
from ..annotations.full_page import FullPage
from ..stitching.stitch_settings import combine_multiple_pages_and_resolve


def _run_split_prediction_job(job: InferenceJob, verbose: bool = False) -> FullPage:
    assert job.split_settings is not None
    assert job.stitch_settings is not None

    image = job.image if isinstance(job.image, np.ndarray) else np.asarray(job.image)
    height, width = image.shape[:2]

    # copy the settings so that tals-based window resizing does not
    # leak into other jobs that share the same SplitSettings instance
    settings = replace(job.split_settings)
    settings.update_window_size_based_on_tals(max(width, height))

    # create splits
    splits = splitting.create_split_box_matrix(
        (width, height),
        window_size=(settings.width, settings.height),
        overlap_ratio=settings.overlap_ratio,
    )
    tiles = splitting.create_split_images(image, splits)

    if verbose:
        print(f"Created total of {len(tiles)} tiles")

    # predict
    subpages: list[FullPage] = job.model_wrapper.predict_multiple(
        tiles, wanted_ids=job.wanted_ids, verbose=verbose, batch_size=job.batch_size
    )

    # resolve
    return combine_multiple_pages_and_resolve(
        subpages,
        splits,
        settings,
        job.stitch_settings,
        verbose=verbose,
    )


def _run_normal_prediction_job(job: InferenceJob, verbose: bool = False) -> FullPage:
    return job.model_wrapper.predict_single(
        job.image, wanted_ids=job.wanted_ids, verbose=verbose
    )


def run_prediction_jobs(
    jobs: Iterable[InferenceJob] | InferenceJob, verbose: bool = False
) -> FullPage:
    """
    Runs given inference jobs and combines all outputs into a single FullPage.

    :param jobs: inference jobs
    :param verbose: make script verbose
    :return: predicted FullPage
    """
    if isinstance(jobs, InferenceJob):
        jobs = [jobs]

    fp: FullPage | None = None
    for job in jobs:
        result = _run_prediction_job(job, verbose=verbose)
        if fp is None:
            fp = result
        else:
            fp.extend_page(result)

    return fp


def _run_prediction_job(job: InferenceJob, verbose: bool = False) -> FullPage:
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

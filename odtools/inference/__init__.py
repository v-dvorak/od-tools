"""
Contains all methods necessary for model inference
and classes for setting up these inferences.
"""

from .inference_job import InferenceJob
from .model_wrappers import IModelWrapper
from .run_job import run_prediction_jobs
from .split_settings import SplitSettings

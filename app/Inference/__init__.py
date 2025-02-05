"""
Contains all methods necessary for model inference
and classes for setting up these inferences.
"""

from .InferenceJob import InferenceJob
from .ModelType import ModelType
from .RunJob import run_prediction_job, run_multiple_prediction_jobs
from .SplitSettings import SplitSettings

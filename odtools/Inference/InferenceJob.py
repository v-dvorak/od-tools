from dataclasses import dataclass
import numpy as np
from typing import Optional
from PIL.ImageFile import ImageFile

from .ModelWrappers import IModelWrapper
from .SplitSettings import SplitSettings


@dataclass(frozen=True)
class InferenceJob:
    """
    Represents an inference job, storing all necessary data for model execution,
    including the image, model, and optional split settings.
    """

    image: np.ndarray | ImageFile
    """The input image, either as a ``PIL ImageFile`` object or a NumPy array."""
    model_wrapper: IModelWrapper
    """The loaded model that will be used for inference."""
    wanted_ids: Optional[list[int]] = None
    """List of class IDs that will be retrieved after inference. Can improve performance."""
    split_settings: Optional[SplitSettings] = None
    """
    Optional settings for splitting the image before inference.
    If provided, the inference will be run in split mode.
    """

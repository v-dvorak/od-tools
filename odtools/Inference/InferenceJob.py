from dataclasses import dataclass
import numpy as np
from typing import Optional
from PIL.ImageFile import ImageFile

from .ModelWrappers import IModelWrapper
from .SplitSettings import SplitSettings
from ..stitching import StitchSettings


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
    stitch_settings: Optional[StitchSettings] = None
    """
    If an image was split into multiple windows before inference,
    these settings are used to put it back together.
    """

    def __post_init__(self) -> None:
        if self.split_settings is not None and self.stitch_settings is None:
            raise ValueError("Stitch settings have to be set, if split setting are")
    
    def report(self) -> str:
        return (
            f"{type(self).__name__} with "
            f"{type(self.model_wrapper).__name__.replace('ModelWrapper', '')}:\n"
            f"\tSplit: {self.split_settings}\n"
            f"\tStitch: {self.stitch_settings}"
        )
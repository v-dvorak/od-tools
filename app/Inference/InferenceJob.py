import numpy as np
from PIL.ImageFile import ImageFile

from .ModelWrappers import IModelWrapper
from .SplitSettings import SplitSettings


class InferenceJob:
    """
    Represents an inference job, storing all necessary data for model execution,
    including the image, model, and optional split settings.

    :ivar image: The input image, either as a ``PIL ImageFile`` object or a NumPy array.
    :ivar model: The loaded model that will be used for inference.
    :ivar wanted_ids: List of class IDs that will be retrieved after inference. Can improve performance.
    :ivar split_settings: Optional settings for splitting the image before inference.
            If provided, the inference will be run in split mode.
    """

    def __init__(
            self,
            image: np.ndarray | ImageFile,
            model_wrapper: IModelWrapper,
            wanted_ids: list[int] = None,
            split_settings: SplitSettings | None = None
    ):
        """
        :param image: input image
        :param model_wrapper: model loaded inside a wrapper for inference
        :param wanted_ids: list of class IDs to retrieve, if None all wil be retrieved
        :param split_settings: if provided, the inference will be run in split mode
        """
        self.image = image
        self.model = model_wrapper
        self.wanted_ids = wanted_ids
        self.split_settings = split_settings

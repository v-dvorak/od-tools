from typing import Any

import numpy as np
from PIL.ImageFile import ImageFile

from .ModelType import ModelType
from .SplitSettings import SplitSettings


class InferenceJob:
    """
    Represents an inference job, storing all necessary data for model execution,
    including the image, model, and optional split settings.

    :ivar image: The input image, either as a ``PIL ImageFile`` object or a NumPy array.
    :ivar model:The loaded machine learning model that will be used for inference.
    :ivar model_type: The type of model, which must correspond to the provided model instance.
    :ivar split_settings: Optional settings for splitting the image before inference.
            If provided, the inference will be run in split mode.
    """
    def __init__(
            self,
            image: np.ndarray | ImageFile,
            model: Any,
            model_type: ModelType = ModelType.YOLO_DETECTION,
            split_settings: SplitSettings | None = None
    ):
        """
        :param image: input image
        :param model: loaded model for inference
        :param model_type: type of the given model, default is ``ModelType.YOLO_DETECTION``
        :param split_settings: if provided, the inference will be run in split mode
        """
        self.image = image
        self.model = model
        self.model_type = model_type
        self.split_settings = split_settings

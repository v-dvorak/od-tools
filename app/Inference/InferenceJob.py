from typing import Any

import numpy as np
from PIL.ImageFile import ImageFile

from .ModelType import ModelType
from .SplitSettings import SplitSettings


class InferenceJob:
    def __init__(
            self,
            image: np.ndarray | ImageFile,
            model: Any,
            model_type: ModelType = ModelType.YOLO_DETECTION,
            split_settings: SplitSettings = None
    ):
        self.image = image
        self.model = model
        self.model_type = model_type
        self.split_settings = split_settings

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from ...Conversions.Annotations.FullPage import FullPage


class IModelWrapper(ABC):
    """
    Base implementation class for model wrappers.
    """
    
    _WRAPPER_NAME = "base"

    @abstractmethod
    def predict_multiple(
        self,
        tiles: list[np.ndarray],
        wanted_ids: Optional[list[int]] = None,
        verbose: bool = False,
        batch_size: int = 16,
    ) -> list[FullPage]:
        pass

    @abstractmethod
    def predict_single(
        self,
        image: np.ndarray,
        wanted_ids: Optional[list[int]] = None,
        verbose: bool = False,
    ) -> FullPage:
        pass

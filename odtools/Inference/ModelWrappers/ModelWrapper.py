from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

from ...Conversions.Annotations.FullPage import FullPage


class IModelWrapper(ABC):
    """
    Base implementation class for model wrappers.
    """

    @abstractmethod
    def predict_multiple(
            self,
            tiles: list[np.ndarray],
            wanted_ids: Optional[list[int]] = None,
            verbose: bool = False
    ) -> list[FullPage]:
        pass

    @abstractmethod
    def predict_single(
            self,
            image: np.ndarray,
            wanted_ids: Optional[list[int]] = None,
            verbose: bool = False
    ) -> FullPage:
        pass

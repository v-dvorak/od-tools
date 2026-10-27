from abc import ABC, abstractmethod
import numpy as np

from ...conversions.annotations.full_page import FullPage


class IModelWrapper(ABC):
    """
    Base implementation class for model wrappers.
    """

    _WRAPPER_NAME = "base"

    @abstractmethod
    def predict_multiple(
        self,
        tiles: list[np.ndarray],
        wanted_ids: list[int] | None = None,
        verbose: bool = False,
        batch_size: int = 16,
    ) -> list[FullPage]:
        pass

    @abstractmethod
    def predict_single(
        self,
        image: np.ndarray,
        wanted_ids: list[int] | None = None,
        verbose: bool = False,
    ) -> FullPage:
        pass

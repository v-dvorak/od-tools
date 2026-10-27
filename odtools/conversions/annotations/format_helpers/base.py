from abc import ABC, abstractmethod
from pathlib import Path


class _HelperBase(ABC):
    @abstractmethod
    def from_file():
        pass

    @abstractmethod
    def to_file():
        pass

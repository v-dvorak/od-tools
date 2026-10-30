from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from ...annotations import AnnotationType
from ..formats import Format

if TYPE_CHECKING:
    from ...annotations.full_page import FullPage


class _HelperBase(ABC):
    FORMAT_TYPE: Format | None = None
    __supports_from_file__: bool = False
    __supports_to_file__: bool = False

    @staticmethod
    @abstractmethod
    def from_file(
        annotation_path: Path,
        image_path: Path,
        class_reference_table: dict[str, int],
        class_output_names: list[str],
        an_type: AnnotationType = AnnotationType.GROUND_TRUTH,
        with_confidence: bool = False,
    ) -> "FullPage":
        pass

    @staticmethod
    @abstractmethod
    def to_file(
        page: "FullPage",
        output_path: Path,
        with_confidence: bool = False,
    ) -> None:
        pass

    @classmethod
    def supports_from_file(cls) -> bool:
        return cls.__supports_from_file__

    @classmethod
    def supports_to_file(cls) -> bool:
        return cls.__supports_to_file__

import numpy as np
import pytest

from odtools.conversions.annotations import Annotation


@pytest.fixture
def make_annotation():
    """Factory for ground-truth Annotations with sensible defaults."""

    def _make(
        class_id: int,
        left: int,
        top: int,
        width: int,
        height: int,
        confidence: float = 1.0,
        mask: np.ndarray | None = None,
        class_name: str | None = None,
    ) -> Annotation:
        return Annotation(
            class_id,
            left,
            top,
            width,
            height,
            mask,
            class_name if class_name is not None else f"class_{class_id}",
            confidence=confidence,
        )

    return _make

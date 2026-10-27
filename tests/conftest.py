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
    ) -> Annotation:
        return Annotation(
            class_id,
            left,
            top,
            width,
            height,
            mask,
            confidence=confidence,
        )

    return _make

from dataclasses import dataclass
import math
import warnings


@dataclass
class SplitSettings:
    """
    Configuration for splitting an image into tiles before inference.

    This class defines the tile size, overlap between tiles, and handling of bounding boxes
    near tile edges.
    """

    width: int | None = None
    """Width of each tile in pixels. Cannot be set together with ``tals``."""

    height: int | None = None
    """Height of each tile in pixels. Cannot be set together with ``tals``."""

    tals: int | None = None
    """
    Number of "tiles along the longer side".
    Number of tiles spanning the longer side of the image.
    If set, ``width`` and ``height`` are computed dynamically.
    """

    overlap_ratio: float = 0.10
    """Minimum horizontal/vertical overlap between tiles when splitting."""

    edge_offset_ratio: float = 0.04
    """Ratio used to compute ``edge_offset``."""

    def __post_init__(self):
        """
        .. note::
            ``width`` and ``height`` cannot be set at the same time as ``tals``. If ``tals`` is provided,
            ``width`` and ``height`` will be automatically calculated and the given values will be overridden
            to ensure the specified number of tiles covers the longer side of the image.
        """
        if self.tals is None:
            if self.width is None:
                self.width = 640
            if self.height is None:
                self.height = 640
        elif self.width is not None or self.height is not None:
            warnings.warn(
                "`width` and `height` are set at the same time as `tals`, "
                "`width` and `height` will be overridden."
            )

    @property
    def edge_offset(self) -> int:
        """
        Determines the distance from tile edges within which bounding boxes are discarded.

        Computed from the *current* tile size, so it stays correct after
        ``update_window_size_based_on_tals`` resizes the tiles.
        """
        if self.width is None or self.height is None:
            raise ValueError(
                "Tile size is not resolved yet: `edge_offset` requires `width` and "
                "`height`, which are computed lazily in `tals` mode (after "
                "`update_window_size_based_on_tals` is called)."
            )
        return round((self.width + self.height) / 2 * self.edge_offset_ratio)

    def update_window_size_based_on_tals(self, longer_side_px: int):
        """
        Updates tile size based on given longer side of an image, only if ``tals`` is set.

        :param longer_side_px: longer side of an image in pixels
        """
        if self.tals is not None:
            tiles_width = math.ceil(
                longer_side_px
                / (self.tals * (1 - self.overlap_ratio) - self.overlap_ratio)
            )
            self.width, self.height = tiles_width, tiles_width

    @classmethod
    def from_json(cls, data: dict) -> "SplitSettings":
        # Construct while using above defined default values
        kwargs = {}

        if (w_size := data.get("window_size")) is not None:
            width, height = w_size
            kwargs["width"] = width
            kwargs["height"] = height

        if data.get("tals") is not None:
            kwargs["tals"] = data["tals"]

        if data.get("overlap_ratio") is not None:
            kwargs["overlap_ratio"] = data["overlap_ratio"]

        if data.get("edge_offset_ratio") is not None:
            kwargs["edge_offset_ratio"] = data["edge_offset_ratio"]

        return cls(**kwargs)

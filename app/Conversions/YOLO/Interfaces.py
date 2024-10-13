class IYOLODetection:
    def __init__(
            self,
            class_id: int,
            x_center: float,
            y_center: float,
            width: float,
            height: float
    ):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.class_id = class_id

    def __str__(self):
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class IYOLOFullPageDetection:
    def __init__(
            self,
            image_size: tuple[int, int],
            annotations: list[IYOLODetection],
    ):
        self.image_size = image_size
        self.annotations = annotations


class IYOLOSegmentation:
    def __init__(
            self,
            class_id: int,
            coordinates: list[tuple[float, float]]
    ):
        self.coordinates = coordinates
        self.class_id = class_id

    def __str__(self):
        coords = " ".join([f"{point[0]:.6f} {point[1]:.6f}" for point in self.coordinates])
        return f"{self.class_id} {coords}"

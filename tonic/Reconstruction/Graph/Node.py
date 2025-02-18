from abc import abstractmethod
from typing import Self, Any

import numpy as np

from .Names import NodeName
from ...Conversions.Annotations.Annotation import Annotation
from ...Conversions.BoundingBox import BoundingBox, Direction


class BaseNode:
    name: NodeName
    total_box: BoundingBox
    _children: list[Self]
    _tags: dict[str, Any]

    def __init__(self, tags: dict[str, Any] = None, name: NodeName = None):
        self.total_bbox = None
        self.name = name
        self._children = []
        self._tags: dict[str, Any] = tags if tags is not None else {}

    def add_child(self, child: Self):
        self._children.append(child)

    def children(self) -> list[Self]:
        return self._children

    def get_tag(self, key: str) -> Any:
        return self._tags.get(key, None)

    def set_tag(self, key: str, value: Any):
        self._tags[key] = value

    @abstractmethod
    def update_total_bbox(self):
        raise NotImplementedError()


class Node(BaseNode):
    """
    Object with a bounding box in a scene.
    """
    annot: Annotation

    def __init__(self, base: Annotation, tags: dict[str, Any] = None, name: NodeName = None):
        super().__init__(tags=tags, name=name)
        self.annot = base
        self.total_bbox = base.bbox

    def add_child(self, child: Self, update_t_bbox: bool = False):
        self._children.append(child)
        if update_t_bbox:
            self.update_total_bbox()

    def update_total_bbox(self):
        if len(self._children) > 0:
            temp_box = BoundingBox(
                min(self._children, key=lambda b: b.annot.bbox.left).annot.bbox.left,
                min(self._children, key=lambda b: b.annot.bbox.top).annot.bbox.top,
                max(self._children, key=lambda b: b.annot.bbox.right).annot.bbox.right,
                max(self._children, key=lambda b: b.annot.bbox.bottom).annot.bbox.bottom
            )
            self.total_bbox = BoundingBox(
                temp_box.left if temp_box.left < self.total_bbox.left else self.total_bbox.left,
                temp_box.top if temp_box.top < self.total_bbox.top else self.total_bbox.top,
                temp_box.right if temp_box.right > self.total_bbox.right else self.total_bbox.right,
                temp_box.bottom if temp_box.bottom > self.total_bbox.bottom else self.total_bbox.bottom
            )


class VirtualNode(BaseNode):
    """
    Virtual object without a bounding box in a scene.
    """
    _children: list[BaseNode]

    def __init__(self, children: list[BaseNode], tags: dict[str, Any] = None, name: NodeName = None):
        super().__init__(tags=tags, name=name)
        self._children: list[BaseNode] = children
        self.update_total_bbox()

    def update_total_bbox(self):
        if len(self._children) > 0:
            self.total_bbox = BoundingBox(
                min(self._children, key=lambda b: b.total_bbox.left).total_bbox.left,
                min(self._children, key=lambda b: b.total_bbox.top).total_bbox.top,
                max(self._children, key=lambda b: b.total_bbox.right).total_bbox.right,
                max(self._children, key=lambda b: b.total_bbox.bottom).total_bbox.bottom
            )

    def children(self) -> list[BaseNode]:
        return self._children


def assign_to_closest(
        target: list[Node],
        source: list[Node],
        upper_limit: float = None,
        verbose: bool = False
):
    """
    Assigns object from source to targets based on their distance from them.
    Modifies the target list in place.

    :param target: list of targets to assign sources to
    :param source: list of sources to assign to targets
    :param upper_limit: maximum distance to assign source to target
    :param verbose: make script verbose
    """
    unable_to_add = 0

    for current_source in source:
        best_distance = np.inf
        best_target: Node = None

        for current_target in target:
            if current_source.annot.bbox.is_fully_inside(current_target.annot.bbox, direction=Direction.HORIZONTAL):
                current_distance = current_target.annot.bbox.center_distance(
                    current_source.annot.bbox,
                    direction=Direction.VERTICAL
                )
                if (upper_limit is None or current_distance < upper_limit) and current_distance < best_distance:
                    best_distance = current_distance
                    best_target = current_target

        if best_target is None:
            if verbose:
                print(
                    "Warning: No suitable target found for source:",
                    f"{current_source.annot.bbox}, id {current_source.annot.class_id}"
                )
            else:
                unable_to_add += 1
        else:
            best_target.add_child(current_source)

    if not verbose and unable_to_add > 0:
        print(f"Warning: No suitable target found for {unable_to_add} source objects")

    for current_target in target:
        current_target.update_total_bbox()


def sort_to_strips_with_threshold(
        nodes: list[Node],
        iou_threshold: float,
        direction: Direction = Direction.HORIZONTAL,
        check_intersections: bool = False
) -> list[list[Node]]:
    """
    Sort objects according to the given threshold into strips (rows/columns).
    HORIZONTAL corresponds to the reading order: left to right, top to bottom.
    VERTICAL corresponds to the reading order: bottom to top, left to right.

    :param nodes: list of objects to sort
    :param iou_threshold: threshold for sorting, how big there could be between two objects in the same strip
    :param direction: the direction of sorting
    :param check_intersections: if true, objects is assigned to strip
    if it intersects with any other of the strip in any direction
    :return: list of sorted strips
    """

    def _intersects_any(row: list[Node], node: Node):
        for n in row:
            if node.annot.bbox.intersects(n.annot.bbox):
                return True
        return False

    if len(nodes) == 0:
        return []

    if direction == Direction.HORIZONTAL:
        top_sorted = sorted(nodes, key=lambda n: n.annot.bbox.top)
        sorted_rows: list[list[Node]] = []
        row: list[Node] = []
        for node in top_sorted:
            if len(row) == 0:
                row.append(node)
                continue

            computed_iou = node.annot.bbox.intersection_over_union(
                row[-1].annot.bbox,
                direction=Direction.VERTICAL
            )
            inter_any = (check_intersections and _intersects_any(row, node))

            if computed_iou > iou_threshold or inter_any:
                row.append(node)
            else:
                sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.left))
                row = [node]

        sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.left))

        return sorted_rows

    elif direction == Direction.VERTICAL:
        left_sorted = sorted(nodes, key=lambda n: n.annot.bbox.left)
        sorted_rows: list[list[Node]] = []
        row: list[Node] = []
        for node in left_sorted:
            if len(row) == 0:
                row.append(node)
                continue

            computed_iou = node.annot.bbox.intersection_over_union(
                row[-1].annot.bbox,
                direction=Direction.HORIZONTAL
            )
            inter_any = (check_intersections and _intersects_any(row, node))

            if computed_iou > iou_threshold or inter_any:
                row.append(node)
            else:
                sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.top, reverse=True))
                row = [node]

        sorted_rows.append(sorted(row, key=lambda n: n.annot.bbox.top, reverse=True))

        return sorted_rows

    else:
        raise NotImplementedError(f"Not implemented for direction {direction}")

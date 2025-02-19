from typing import Any

from .Graph.Names import NodeName
from .Graph.Node import Node
from odtools.Conversions.Annotations.Annotation import Annotation


def _preprocess_annots_without_tags(node_name: NodeName, annots: list[Annotation]) -> list[Node]:
    return [Node(annot, name=node_name) for annot in annots]


def _preprocess_annots_with_tags(
        node_name: NodeName,
        annots: list[Annotation],
        tags: list[tuple[str, Any]]
) -> list[Node]:
    output: list[Node] = []

    for annot in annots:
        node = Node(annot, name=node_name)
        for tag_name, tag_value in tags:
            node.set_tag(tag_name, tag_value)
        output.append(node)

    return output


def preprocess_annots_for_reconstruction(
        data: list[
            tuple[NodeName, list[str], list[tuple[list[Annotation], list[Any]]]]
            | tuple[NodeName, list[Annotation]]]) -> list[list[Node]]:
    """
    Takes a list of nodes names and tags, processes them and returns accordingly tagged nodes.

    The input format is a list of two types of specifications:
        - ``tuple[NodeName, list[Annotation]]``: every annotation in the given list will be turned into a ``Node``
            with this name
        - ``tuple[NodeName, list[str], list[tuple[list[Annotation], list[Any]]]]``:
            every annotation in the given list will be turned into a ``Node`` with this name,
            also it will be tagged with the tags type given by the ``list[str]``
            and tags' values will be set to values specified by the ``list[Any]``


    :param data: list of preprocessing specifications and data
    :return: list of lists of processed nodes with respective tags

    Example usage::

        preprocess_annots_for_reconstruction(
            [
                (
                    NodeName.ACCIDENTAL, [ACCIDENTAL_TYPE_TAG],
                    [
                        (combined.annotations[-3], [AccidentalType.FLAT]),
                        (combined.annotations[-2], [AccidentalType.NATURAL]),
                        (combined.annotations[-1], [AccidentalType.SHARP])
                    ]
                ),
                (
                    NodeName.GRAND_STAFF, combined.annotations[4]
                )
            ]
        )

    """
    output: list[list[Node]] = []

    for dato in data:
        if len(dato) == 2:
            dato: tuple[NodeName, list[Annotation]]
            name, annots = dato
            output.append(_preprocess_annots_without_tags(name, annots))

        elif len(dato) == 3:
            dato: tuple[NodeName, list[str], list[tuple[list[Annotation], list[Any]]]]
            group = []

            name, tag_names, values = dato
            for annots, tag_values in values:
                group.extend(_preprocess_annots_with_tags(name, annots, list(zip(tag_names, tag_values))))

            output.append(group)
        else:
            raise ValueError("Unsupported format")

    return output

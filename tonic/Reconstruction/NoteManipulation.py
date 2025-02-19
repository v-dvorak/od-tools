from pathlib import Path

import numpy as np

from .Graph.Node import Node, assign_to_closest
from .Graph.Tags import SYMBOL_PITCH_TAG, SYMBOL_GS_INDEX_TAG
from .VizUtils import write_note_heights_to_image
from odtools.Splitting import draw_rectangles_on_image


def _compute_note_pitches(measure: Node):
    """
    Computes each note's distance from the bottom staff line and assigns a pitch
    for all notes in the given measure.
    """
    half_line_height = measure.annot.bbox.height / 8
    for note in measure.children():
        x_center, _ = note.annot.bbox.center()
        distance_from_zero = measure.annot.bbox.bottom - x_center
        note.set_tag(SYMBOL_PITCH_TAG, distance_from_zero / half_line_height)


def _assign_gs_index_to_notes(measures: list[Node], gs_index: int):
    for measure in measures:
        for note in measure.children():
            note.set_tag(SYMBOL_GS_INDEX_TAG, gs_index)


def assign_notes_to_measures_and_compute_pitch(
        measures: list[Node],
        notes: list[Node],
        ual_factor: float,
        image_path: Path = None,
        verbose: bool = False,
        visualize: bool = False,
) -> None:
    """
    Assigns given notes to given measures and computes their pitches.
    The method modifies given list of measures in place.
    Note are assigned to measures based on distance between their centers and the upper assignment limit (UAL).

    If the note's distance from the closest measure is larger than UAL, the note is dropped from further processing.

    UAL is computed as: `np.mean(heights of all measures) * ual_factor`

    For example, `ual_factor` set to 1.5 means that notes that are
    "one measure height above" and "one measure height below" are considered valid.

    Pitch is computed based on notes distance from the bottom staff line of the measure is it assigned to
    and the average distance between staff lines in that particular measure.

    :param measures: list of measures represented as Node
    :param notes: list of notes represented as Node
    :param ual_factor: controls how fat a note can be from center of measure to still be considered a valid note
    :param image_path: path to image for visualization
    :param verbose: make script verbose
    :param visualize: show visualization on screen
    """
    if image_path is None and visualize:
        raise ValueError("Image path is required when visualize is set to True.")

    upper_assignment_limit = np.mean([m.annot.bbox.height for m in measures]) * ual_factor
    assign_to_closest(measures, notes, upper_limit=upper_assignment_limit, verbose=verbose)

    # assign pitch to each note
    for measure in measures:
        _compute_note_pitches(measure)

    if visualize:
        print("Showing note pitches...")
        measure_height_viz = draw_rectangles_on_image(
            image_path,
            [m.total_bbox for m in measures]
        )
        write_note_heights_to_image(
            measure_height_viz,
            [n for m in measures for n in m.children()]
        )
        input("Press Enter to continue")

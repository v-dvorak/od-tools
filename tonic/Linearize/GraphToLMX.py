from decimal import Decimal, ROUND_HALF_UP

from .Tokens import (G_CLEF_ZERO_PITCH_INDEX, F_CLEF_ZERO_PITCH_INDEX, PITCH_TOKENS, NOTE_TOKEN, CHORD_TOKEN,
                     GS_CLEF_BIG_TOKEN, BASE_TIME_BEAT_BIG_TOKEN, STAFF_TOKEN, DEFAULT_STEM_TOKEN,
                     MEASURE_TOKEN,
                     DEFAULT_KEY_TOKEN)
from ..Linearize.LMXWrapper import LMXWrapper
from ..Reconstruction.Graph.Names import NodeName
from ..Reconstruction.Graph.Node import Node, VirtualNode
from ..Reconstruction.Graph.Tags import (NOTEHEAD_TYPE_TAG, ACCIDENTAL_TYPE_TAG, SYMBOL_GS_INDEX_TAG, SYMBOL_PITCH_TAG)


def _symbol_pitch_to_str(note: Node) -> int:
    # skip python default rounding (0.5 should be rounded to 1)
    return int(Decimal(note.get_tag(SYMBOL_PITCH_TAG)).to_integral(ROUND_HALF_UP))


def _notehead_to_string(note: Node) -> str:
    gs_index = note.get_tag(SYMBOL_GS_INDEX_TAG)
    pitch = _symbol_pitch_to_str(note)
    if gs_index is not None:
        return f"{gs_index}{note.get_tag(NOTEHEAD_TYPE_TAG)}{pitch}"
    else:
        return f"{note.get_tag(NOTEHEAD_TYPE_TAG)}{pitch}"


def _accident_to_string(note: Node) -> str:
    gs_index = note.get_tag(SYMBOL_GS_INDEX_TAG)
    pitch = _symbol_pitch_to_str(note)
    if gs_index is not None:
        return f"{gs_index}{note.get_tag(ACCIDENTAL_TYPE_TAG)}{pitch}"
    else:
        return f"{note.get_tag(ACCIDENTAL_TYPE_TAG)}{pitch}"


def symbol_to_str(note: Node) -> str:
    match note.name:
        case NodeName.NOTEHEAD:
            return _notehead_to_string(note)
        case NodeName.ACCIDENTAL:
            return _accident_to_string(note)
        case _:
            raise ValueError(f"Unknown symbol type {note.name}")


def get_note_pitch(note: Node) -> str:
    gs_index = note.get_tag(SYMBOL_GS_INDEX_TAG)
    pitch = round(note.get_tag(SYMBOL_PITCH_TAG))

    if gs_index is None or gs_index == 1:
        pitch_index = G_CLEF_ZERO_PITCH_INDEX + pitch
    elif gs_index == 2:
        pitch_index = F_CLEF_ZERO_PITCH_INDEX + pitch
    else:
        raise ValueError(f"Unknown value of {SYMBOL_GS_INDEX_TAG}: {gs_index}")

    return PITCH_TOKENS[pitch_index]


def _note_to_lmx(note: Node) -> list[str]:
    gs_tag = note.get_tag(SYMBOL_GS_INDEX_TAG)
    pitch_token = get_note_pitch(note)

    return [
        pitch_token,
        NOTE_TOKEN,
        DEFAULT_STEM_TOKEN,
        f"{STAFF_TOKEN}:{gs_tag if gs_tag is not None else 1}"
    ]


def _linearize_note_event_to_lmx(event: VirtualNode) -> list[str]:
    sequence: list[str] = []
    first = True
    for note in event.children():
        note: Node

        if first:
            first = False
        else:
            sequence.append(CHORD_TOKEN)

        sequence.extend(_note_to_lmx(note))

    return sequence


def linearize_note_events_to_lmx(measure_groups: list[list[VirtualNode]]) -> LMXWrapper:
    note_written = False
    sequence: list[str] = []
    first = True
    for row in measure_groups:

        for measure in row:

            sequence.append(MEASURE_TOKEN)
            if first:
                sequence.append(DEFAULT_KEY_TOKEN)
                sequence.extend(BASE_TIME_BEAT_BIG_TOKEN.split())
                sequence.extend(GS_CLEF_BIG_TOKEN.split())
                first = False

            for child in measure.children():
                child: VirtualNode
                if child.name == NodeName.NOTE_EVENT:
                    sequence.extend(_linearize_note_event_to_lmx(child))
                    note_written = True

    if note_written:
        return LMXWrapper(sequence)
    else:
        print("Warning: No note events were written.")
        return LMXWrapper(sequence)

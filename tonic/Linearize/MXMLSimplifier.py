from pathlib import Path

import smashcima as sc
from smashcima import Clef, Event, Note, Score, StaffSemantic

from .Tokens import G_CLEF_ZERO_PITCH_INDEX, F_CLEF_ZERO_PITCH_INDEX
from .Tokens import (NOTE_TOKEN, CHORD_TOKEN, GS_CLEF_TOKEN, BASE_TIME_BEAT_TOKEN, STAFF_TOKEN, MEASURE_TOKEN,
                     DEFAULT_KEY_TOKEN, DEFAULT_STEM_TOKEN, PITCH_TOKENS)
from .lmx_to_musicxml import lmx_to_musicxml


def _get_note_relative_pitch_to_first_staff_line(note: Note) -> int:
    event = Event.of_durable(note)
    staff_sem = StaffSemantic.of_durable(note)
    clef: Clef = event.attributes.clefs[staff_sem.staff_number]

    # get absolute position of notehead on staff
    pitch_position = clef.pitch_to_pitch_position(note.pitch) + 4
    # +4 -> smashcima indexes from the middle staff line, this project indexes from the bottom staff line

    return pitch_position


def _note_to_lmx(note: Note) -> str:
    # get absolute position of notehead on staff
    pitch_position = _get_note_relative_pitch_to_first_staff_line(note)

    # get staff index grand staff
    staff_index = StaffSemantic.of_durable(note).staff_number

    # simplify note pitch: G clef at first staff, F clef at second staff
    if staff_index == 1:
        pitch_index = G_CLEF_ZERO_PITCH_INDEX + pitch_position
    elif staff_index == 2:
        pitch_index = F_CLEF_ZERO_PITCH_INDEX + pitch_position
    else:
        raise NotImplementedError(f"Unsupported staff index \"{staff_index}\"")

    return " ".join([PITCH_TOKENS[pitch_index], NOTE_TOKEN, DEFAULT_STEM_TOKEN,
                     f"{STAFF_TOKEN}:{staff_index}"])


def _event_to_lmx(event: Event) -> list[str]:
    sequence: list[str] = []
    is_chord = False
    notes = [durable for durable in event.durables if isinstance(durable, Note)]
    notes: list[Note]
    notes = sorted(notes, key=lambda n: n.pitch.get_linear_pitch())
    for note in notes:
        if isinstance(note, Note):
            if is_chord:
                sequence.append(CHORD_TOKEN)
            sequence.append(_note_to_lmx(note))
            is_chord = True

    return sequence


def scene_to_lmx(score: Score) -> str:
    sequence: list[str] = []

    sequence.append(MEASURE_TOKEN)
    sequence.append(DEFAULT_KEY_TOKEN)
    sequence.append(BASE_TIME_BEAT_TOKEN)
    sequence.append(GS_CLEF_TOKEN)
    first = True
    for part in score.parts:
        for measure in part.measures:
            if not first:
                sequence.append(MEASURE_TOKEN)
            first = False
            for event in measure.events:
                sequence.extend(_event_to_lmx(event))

    return " ".join(sequence)


def complex_musicxml_file_to_lmx(file_path: Path) -> str:
    score = sc.loading.load_score(file_path)
    return scene_to_lmx(score)


def simplify_musicxml_file(input_path: Path, output_path: Path):
    output_lmx = complex_musicxml_file_to_lmx(input_path)
    output_xml = lmx_to_musicxml(output_lmx)

    with open(output_path, "w", encoding="utf8") as f:
        f.write(output_xml)

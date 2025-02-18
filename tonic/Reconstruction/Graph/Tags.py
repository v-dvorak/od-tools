from enum import Enum

# NODE TAGS
SYMBOL_PITCH_TAG = "pitch"
SYMBOL_GS_INDEX_TAG = "gs_index"

NOTEHEAD_TYPE_TAG = "notehead_type"
ACCIDENTAL_TYPE_TAG = "accidental_type"


class NoteheadType(Enum):
    HALF = 0
    FULL = 1

    def __str__(self) -> str:
        match self:
            case NoteheadType.HALF:
                return "h"
            case NoteheadType.FULL:
                return "f"
            case _:
                raise ValueError("Unknown notehead type")


class AccidentalType(Enum):
    SHARP = 0
    FLAT = 1
    NATURAL = 2

    def __str__(self) -> str:
        match self:
            case AccidentalType.SHARP:
                return "#"
            case AccidentalType.FLAT:
                return "b"
            case AccidentalType.NATURAL:
                return "N"
            case _:
                raise ValueError("Unknown accidental type")

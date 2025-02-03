from enum import Enum

NOTE_PITCH_TAG = "pitch"
NOTE_GS_INDEX_TAG = "gs_index"
NOTEHEAD_TYPE_TAG = "notehead_type"


class NoteheadType(Enum):
    HALF = 0
    FULL = 1

    def __str__(self) -> str:
        match self:
            case NoteheadType.FULL:
                return "f"
            case NoteheadType.HALF:
                return "h"
            case _:
                raise ValueError("Unknown notehead type")

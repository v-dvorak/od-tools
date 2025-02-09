from enum import Enum


class NodeName(Enum):
    """
    Names used to denote specific nodes in the graph.

    Names that start with "_" should never leak through the algorithms to the end user.
    """
    # basic
    NOTEHEAD = "notehead"
    ACCIDENTAL = "accidental"
    MEASURE = "measure"
    GRAND_STAFF = "grand_staff"

    # grouping
    SYMBOL_GROUP = "symbol_group"
    NOTE_EVENT = "note_event"

    _MEASURER_GROUP = "measure_group"

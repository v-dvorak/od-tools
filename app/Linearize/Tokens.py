from lmx.linearization.vocabulary import PITCH_TOKENS

BASE_TIME_BEAT_TOKEN = "time beats:4 beat-type:4"
GS_CLEF_TOKEN = "clef:G2 staff:1 clef:F4 staff:2"
DEFAULT_KEY_TOKEN = "key:fifths:0"

INDENTATION = 4 * " "

G_CLEF_ZERO_PITCH_INDEX = PITCH_TOKENS.index("E4")
F_CLEF_ZERO_PITCH_INDEX = PITCH_TOKENS.index("G2")

CHORD_TOKEN = "chord"
NOTE_TOKEN = "quarter"
STAFF_TOKEN = "staff"
MEASURE_TOKEN = "measure"

_STEM_TOKEN = "stem"
DEFAULT_STEM_TOKEN = f"{_STEM_TOKEN}:up"

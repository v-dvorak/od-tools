import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Self

from lmx.linearization.Delinearizer import Delinearizer
from lmx.linearization.Linearizer import Linearizer
from lmx.symbolic.MxlFile import MxlFile
from lmx.symbolic.part_to_score import part_to_score
from nltk.metrics import edit_distance

from .Tokens import (MEASURE_TOKEN, DEFAULT_KEY_TOKEN,
                     TIME_TOKEN, CLEF_G2_TOKEN, CHORD_TOKEN, NOTE_QUARTER_TOKEN,
                     DEFAULT_STEM_TOKEN, STAFF_TOKEN, PITCH_TOKENS)


class LMXWrapper:
    _CHORD_PITCH_GENERAL_TOKEN = "chord_pitch"
    _note_attribute_order = {
        CHORD_TOKEN: 0,
        _CHORD_PITCH_GENERAL_TOKEN: 1,
        NOTE_QUARTER_TOKEN: 2,
        DEFAULT_STEM_TOKEN: 3,
        STAFF_TOKEN: 4
    }

    def __init__(self, tokens: list[str]):
        self.tokens = tokens

    @classmethod
    def from_lmx_string(cls, score: str) -> Self:
        """
        Creates a LMXWrapper from a LMX string.

        :param score: LMX string
        :return: LMXWrapper
        """
        return LMXWrapper(score.split())

    @staticmethod
    def _mxl_to_tokens(mxl: MxlFile) -> list[str]:
        try:
            part = mxl.get_piano_part()
        except:
            part = mxl.tree.find("part")

        if part is None or part.tag != "part":
            print("No <part> element found.")
            exit()

        linearizer = Linearizer()
        linearizer.process_part(part)
        return linearizer.output_tokens

    @classmethod
    def from_musicxml_file(cls, musicxml_file: Path) -> Self:
        """
        Loads MusicXML file and returns a LMXWrapper object.

        :param musicxml_file: path to MusicXML file
        :return: LMXWrapper
        """
        with open(musicxml_file, "r") as f:
            input_xml = f.read()
            mxl = MxlFile(ET.ElementTree(
                ET.fromstring(input_xml))
            )
        return LMXWrapper(LMXWrapper._mxl_to_tokens(mxl))

    @staticmethod
    def normalized_levenstein_distance(predicted: "LMXWrapper", ground_truth: "LMXWrapper") -> float:
        """
        Returns the normalized Levenstein distance between the tokens
        of the predicted and ground truth LMXWrapper instances.

        The total Levenstein distance is divided by the number of tokens in ground truth.

        :param predicted: predicted LMX
        :param ground_truth: ground truth LMX
        :return: normalized Levenstein distance
        """
        return 1 - edit_distance(predicted.tokens, ground_truth.tokens) / len(ground_truth.tokens)

    def to_str(self) -> str:
        return " ".join(self.tokens)

    def __str__(self) -> str:
        return self.to_str()

    def to_musicxml(self) -> str:
        dl = Delinearizer()
        dl.process_text(self.to_str())
        score_etree = part_to_score(dl.part_element)
        output_xml = str(
            ET.tostring(
                score_etree.getroot(),
                encoding="utf-8",
                xml_declaration=True
            ), "utf-8")

        return output_xml

    def standardize(self) -> None:
        dl = Delinearizer()
        dl.process_text(self.to_str())
        score_tree = part_to_score(dl.part_element)

        mxl = MxlFile(score_tree)
        self.tokens = LMXWrapper._mxl_to_tokens(mxl)

    def to_human_readable(self, indent: int = 4) -> str:
        # for the simplified format, minimal number of tokens is nine
        #  1 | measure
        # +1 |     key:fifths:0
        # +3 |     time beats:4 beat-type:4
        # +4 |     clef:G2 staff:1 clef:F4 staff:2
        # =9
        if len(self.tokens) > 8:
            output = ""
            indent_whitespace = indent * " "
            token_index = 0
            note_line: list[str] = []
            reset_note_line = False  # reset note construction if non-note tokens  are met
            last_note_attribute = None

            def _reset_note_constr():
                nonlocal note_line, output
                if note_line[0] == CHORD_TOKEN:
                    output += indent_whitespace + " ".join(note_line) + "\n"
                else:
                    output += indent_whitespace + (len(CHORD_TOKEN) + 1) * " " + " ".join(note_line) + "\n"
                note_line = []

            def _pad_indentation_and_new_line(tokens: list[str]) -> str:
                nonlocal indent_whitespace
                return indent_whitespace + " ".join(tokens) + "\n"

            while token_index < len(self.tokens):
                current_token = self.tokens[token_index]
                # measure separator
                if current_token == MEASURE_TOKEN:
                    output += current_token + "\n"
                    reset_note_line = True
                    token_index += 1
                # FILE HEADER
                elif current_token == DEFAULT_KEY_TOKEN:
                    output += _pad_indentation_and_new_line([current_token])
                    reset_note_line = True
                    token_index += 1
                elif current_token == TIME_TOKEN:
                    output += _pad_indentation_and_new_line([
                        current_token,  # time
                        self.tokens[token_index + 1],  # beats:4
                        self.tokens[token_index + 2]  # beat-type:4
                    ])
                    reset_note_line = True
                    token_index += 3
                elif current_token == CLEF_G2_TOKEN:
                    output += _pad_indentation_and_new_line([
                        current_token,  # clef:G2
                        self.tokens[token_index + 1],  # staff:1
                        self.tokens[token_index + 2],  # clef:F4
                        self.tokens[token_index + 3],  # staff:2
                    ])
                    reset_note_line = True
                    token_index += 4
                # NOTE LINE
                else:
                    # construct note line based in current token
                    # pitches
                    if current_token in PITCH_TOKENS:
                        attr_token = self._CHORD_PITCH_GENERAL_TOKEN
                    # "chord", "quarter", ...
                    elif current_token in [CHORD_TOKEN, NOTE_QUARTER_TOKEN, DEFAULT_STEM_TOKEN]:
                        attr_token = current_token
                    # staff
                    elif current_token.startswith(STAFF_TOKEN):
                        attr_token = STAFF_TOKEN
                    else:
                        raise ValueError(f"Invalid token \"{current_token}\"")

                    if len(note_line) > 0:
                        assert last_note_attribute is not None
                        # check the order of last and current token,
                        # note line should be reset if start of other note line is met
                        if self._note_attribute_order[attr_token] < self._note_attribute_order[last_note_attribute]:
                            _reset_note_constr()

                    note_line.append(current_token)
                    last_note_attribute = attr_token
                    token_index += 1

                if reset_note_line and len(note_line) > 0:
                    _reset_note_constr()
                reset_note_line = False

            # dump rest of constructed note line
            if len(note_line) > 0:
                _reset_note_constr()

            return output

        # invalid format
        raise ValueError(f"Invalid input sequence, to few tokens {len(self.tokens)}")

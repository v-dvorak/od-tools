import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Self

from lmx.linearization.Delinearizer import Delinearizer
from lmx.linearization.Linearizer import Linearizer
from lmx.symbolic.MxlFile import MxlFile
from lmx.symbolic.part_to_score import part_to_score
from nltk.metrics import edit_distance


class LMXWrapper:
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

import xml.etree.ElementTree as ET

from lmx.linearization.Delinearizer import Delinearizer
from lmx.symbolic.part_to_score import part_to_score


def lmx_to_musicxml(linearized: str) -> str:
    ln = Delinearizer()

    ln.process_text(linearized)

    score_etree = part_to_score(ln.part_element)
    output_xml = str(
        ET.tostring(
            score_etree.getroot(),
            encoding="utf-8",
            xml_declaration=True
        ), "utf-8")

    return output_xml

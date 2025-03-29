from enum import Enum
from typing import Self

from prettytable import PrettyTable, MARKDOWN


def get_mapping_and_names_from_config(config: dict, verbose: bool = False) -> tuple[dict[str, int], list[str]]:
    """
    Get class name mapping to ids and output names from config file.

    :param config: loaded config dictionary
    :param verbose: make script verbose
    :return: mapping of class names to class id and class output names
    """
    if verbose:
        print("Parsing config...")
        print()

    output_names = []
    class_id_reference_table = {}
    class_mapping = config["class_mapping"]

    if verbose:
        # setup table
        table = PrettyTable(["Input name", "Output name", "ID"])
        table.set_style(MARKDOWN)
        table.align["Input name"] = "l"
        table.align["Output name"] = "l"
        table.align["ID"] = "c"

    for class_id in range(len(class_mapping)):
        dato = class_mapping[str(class_id)]
        if isinstance(dato, list):
            if isinstance(dato[0], list):
                # format: "class_id" -> [ [ "class1", "class2", ... ] , "output_name" ]
                group, output_name = dato[0], dato[1]
            else:
                # format: "class_id" -> ["class1", "class2", ... ]
                group, output_name = dato, dato[0]
        else:
            # format: "class_id" -> "class1"
            group, output_name = [dato], dato

        for name in group:
            class_id_reference_table[name] = class_id
            if verbose:
                table.add_row(['"' + name + '"', '"' + output_name + '"', class_id])

        output_names.append(output_name)

    if verbose:
        table.sortby = "ID"
        print(table)

    return class_id_reference_table, output_names


class ExtendedEnum(Enum):
    @classmethod
    def get_all(cls) -> list[Self]:
        return [e for e in cls]

    @classmethod
    def get_all_value(cls) -> list[int | str]:
        return [e.value for e in cls]

    @classmethod
    def from_string(cls, token: str, lower: bool = True, from_name: bool = False) -> Self:
        if lower:
            token = token.lower()

        if from_name:
            for e in cls:
                if e.name == token:
                    return e
        else:
            for e in cls:
                if e.value == token:
                    return e

        raise ValueError(f"Invalid input format: \"{token}\"")

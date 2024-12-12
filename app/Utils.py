from enum import Enum
from typing import Self


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
        header = f"{' Input name':<33} {' Output name':<30} (ID)"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

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
                q_name = '"' + name + '"'
                q_output_name = '"' + output_name + '"'
                print(f"{q_name:<30} -> {q_output_name:<30} ({class_id})")

        output_names.append(output_name)

    return class_id_reference_table, output_names


class ExtendedEnum(Enum):
    @classmethod
    def get_all(cls) -> list[Self]:
        return [e for e in cls]

    @classmethod
    def get_all_value(cls) -> list[int | str]:
        return [e.value for e in cls]

    @classmethod
    def from_string(cls, token:str) -> Self:
        token = token.lower()
        for e in cls:
            if e.value == token:
                return e

        return ValueError(f"Invalid input format: \"{token}\"")

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
                print(f"Mapping \"{name}\" -> \"{output_name}\" (id = {class_id})")

        output_names.append(output_name)

    if verbose:
        print()
        names = ", ".join(['"' + name + '"' for name in output_names])
        print(f"Output names: {names}")

    return class_id_reference_table, output_names

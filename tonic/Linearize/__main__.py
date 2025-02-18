from argparse import ArgumentParser
from pathlib import Path

from .MXMLSimplifier import simplify_musicxml_file


def main():
    parser = ArgumentParser(

    )
    subparsers = parser.add_subparsers(dest="command", help="Jobs")
    simp_parser = subparsers.add_parser("simplify")

    simp_parser.add_argument("input", help="Path to input file")
    simp_parser.add_argument("-o", "--output", help="Path to output file")

    args = parser.parse_args()

    if args.command == "simplify":

        input_file = Path(args.input)
        if args.output is None:
            output_file = input_file.parent / (input_file.stem + "_simple" + input_file.suffix)
        else:
            output_file = Path(args.output)

        simplify_musicxml_file(input_file, output_file)
        print(f"Saved at: {output_file.absolute()}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
Command-line script to process text files
"""

import sys
import argparse

from typing import List

from pii_manager import VERSION
from pii_manager.api import process_file


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Perform PII processing on a file (version {VERSION})"
    )

    g0 = parser.add_argument_group("Input/output paths")
    g0.add_argument("infile", help="source file")
    g0.add_argument("outfile", help="destination file")

    g1 = parser.add_argument_group("Language specification")
    g1.add_argument("--lang", help="document language", required=True)
    g1.add_argument("--country", nargs="+", help="countries to use")

    g2 = parser.add_argument_group("Task specification")
    g21 = g2.add_mutually_exclusive_group(required=True)
    g21.add_argument("--tasks", nargs="+", help="pii tasks to include")
    g21.add_argument(
        "--all-tasks", action="store_true", help="add all pii tasks available"
    )
    g21.add_argument(
        "--taskfile", nargs="+", help="add all pii tasks defined in a JSON file"
    )

    g3 = parser.add_argument_group("Processing")
    g3.add_argument(
        "--split",
        choices=("line", "sentence", "block"),
        default="line",
        help="document splitting mode (default: %(default)s)",
    )
    g3.add_argument(
        "--mode",
        choices=("replace", "tag", "extract", "full"),
        default="replace",
        help="processing mode (default: %(default)s)",
    )
    g3.add_argument("--template", help="for modes replace & tag, use a custom template")

    g3 = parser.add_argument_group("Other")
    g3.add_argument("--show-stats", action="store_true", help="show statistics")
    g3.add_argument("--show-tasks", action="store_true", help="show defined tasks")

    return parser.parse_args(args)


def main(args: List[str] = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args = vars(args)
    process_file(args.pop("infile"), args.pop("outfile"), args.pop("lang"), **args)


if __name__ == "__main__":
    main()

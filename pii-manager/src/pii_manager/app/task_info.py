"""
Command-line script to show information about available tasks
"""

import sys
import argparse

from typing import List, TextIO


from pii_manager import VERSION
from pii_manager.api import PiiManager



def print_tasks(proc: PiiManager, out: TextIO):
    print(f". Installed tasks [language={proc.lang}]", file=out)
    for (pii, country), doc in proc.task_info().items():
        print(f" {pii.name}  [country={country}]\n   ", doc, file=out)


def process(lang: str, country: List[str] = None,
            tasks: List[str] = None, all_tasks: bool = False,
            debug: bool = False):
    """
    Process the request: show task info
    """
    # Create the object
    proc = PiiManager(lang, country, tasks, all_tasks=all_tasks,
                      debug=debug)
    print_tasks(proc, sys.stdout)


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Show information about available PII tasks (version {VERSION})")

    g1 = parser.add_argument_group("Language specification")
    g1.add_argument("--lang", help="language to load", required=True)
    g1.add_argument("--country", nargs="+", help="countries to use (use 'all' for all countries defined for the language)")

    g2 = parser.add_argument_group("Task specification")
    g21 = g2.add_mutually_exclusive_group(required=True)
    g21.add_argument("--tasks", metavar="TASK_NAME", nargs="+",
                     help="pii tasks to include")
    g21.add_argument("--all-tasks", action="store_true",
                     help="add all pii tasks available")

    g3 = parser.add_argument_group("Other")
    g3.add_argument("--debug", action="store_true", help="debug mode")

    return parser.parse_args(args)


def main(args: List[str] = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    args = vars(args)
    process(args.pop("lang"), **args)


if __name__ == "__main__":
    main()

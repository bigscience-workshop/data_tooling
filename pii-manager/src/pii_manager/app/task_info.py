"""
Command-line script to show information about available tasks
"""

import sys
import argparse

from typing import List, TextIO


from pii_manager import VERSION
from pii_manager.api import PiiManager
from pii_manager.api.file import add_taskfile
from pii_manager.helper.taskdict import language_list, country_list


def print_tasks(proc: PiiManager, out: TextIO):
    print(f". Installed tasks [language={proc.lang}]", file=out)
    for (pii, country), tasklist in proc.task_info().items():
        print(f"\n {pii.name}  [country={country}]   ", file=out)
        for name, doc in tasklist:
            print(f"     {name}: {doc}", file=out)


def process(
    lang: str,
    country: List[str] = None,
    tasks: List[str] = None,
    all_tasks: bool = False,
    taskfile: List[str] = None,
    debug: bool = False,
    **kwargs,
):
    """
    Process the request: show task info
    """
    # Create the object
    proc = PiiManager(lang, country, tasks, all_tasks=all_tasks, debug=debug)
    if taskfile:
        add_taskfile(taskfile, proc)

    # Show tasks
    print_tasks(proc, sys.stdout)


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Show information about available PII tasks (version {VERSION})"
    )

    g1 = parser.add_argument_group("Language specification")
    g11 = g1.add_mutually_exclusive_group(required=True)
    g11.add_argument("--lang", help="language to load")
    g11.add_argument(
        "--list-languages", action="store_true", help="List all defined languages "
    )
    g1.add_argument(
        "--country",
        nargs="+",
        help="countries to use (use 'all' for all countries defined for the language)",
    )

    g2 = parser.add_argument_group("Task specification")
    g21 = g2.add_mutually_exclusive_group()
    g21.add_argument(
        "--tasks", metavar="TASK_NAME", nargs="+", help="pii tasks to include"
    )
    g21.add_argument(
        "--all-tasks", action="store_true", help="add all pii tasks available"
    )
    g2.add_argument(
        "--taskfile", nargs="+", help="add all pii tasks defined in a JSON file"
    )

    g3 = parser.add_argument_group("Other")
    g3.add_argument("--debug", action="store_true", help="debug mode")

    parsed = parser.parse_args(args)
    if not (
        parsed.list_languages or parsed.tasks or parsed.all_tasks or parsed.taskfile
    ):
        print(". Warning: no task list selected")
    return parsed


def main(args: List[str] = None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.list_languages:
        for lang in language_list():
            print(f"  {lang}:", " ".join(country_list(lang)))
    else:
        args = vars(args)
        process(args.pop("lang"), **args)


if __name__ == "__main__":
    main()

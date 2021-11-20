"""
Traverse all folders and gather all implemented PiiTasks into a nested
dictionary

Each dictionary value contains a 3-4 element tuple:
 * lang
 * country
 * PiiEnum
 * task implementation
 * (for regex tasks) task documentation
"""

import sys
import importlib
from pathlib import Path

from typing import Dict, List, Tuple
from types import ModuleType

from pii_manager import PiiEnum
from .exception import InvArgException

# Folder for language-independent tasks
TASK_ANY = "any"

# Name of the list that holds the pii tasks at each module
_LISTNAME = "PII_TASKS"

# --------------------------------------------------------------------------


_LANG = Path(__file__).parents[1] / "lang"


def build_subdict(task_list: List[Tuple], lang: str = None,
                  country: str = None) -> Dict:
    """
    Given a list of task tuples, build the task dict for them
    """
    subdict = {}
    for task in task_list:
        # Checks
        if not isinstance(task, tuple):
            raise InvArgException("Error in tasklist for lang={}, country={}: element is not a tuple", lang, country)
        if not isinstance(task[0], PiiEnum):
            raise InvArgException("Error in tasklist for lang={}, country={}: need a PiiEnum in the first tuple element", lang, country)
        # Add to dict
        subdict[task[0].name] = (lang, country, *task)
    return subdict


def _gather_piitasks(pkg: ModuleType, path: str, lang: str, country: str,
                     debug: bool = False) -> List[Tuple]:
    """
    Import and load all tasks defined in a module
    """
    # Get the list of Python files in the module
    modlist = (
        m.stem
        for m in Path(path).iterdir()
        if m.suffix == ".py" and m.stem != "__init__"
    )

    # Get all tasks defined in those files
    pii_tasks = {}
    for mname in modlist:
        mod = importlib.import_module("." + mname, pkg)
        task_list = getattr(mod, _LISTNAME, None)
        if task_list:
            pii_tasks.update(build_subdict(task_list, lang, country))

    # If debug mode is on, print out the list
    if debug:
        if not pii_tasks:
            print(".. NO PII TASKS for", pkg, file=sys.stderr)
        else:
            print(".. PII TASKS for", pkg, file=sys.stderr)
            print(".. path =", path, file=sys.stderr)
            for task_name, task in pii_tasks.items():
                print("  ", task_name, "->", task[3], file=sys.stderr)

    return pii_tasks


def import_processor(lang: str, country: str = None, debug: bool = False) -> Dict:
    """
    Import all task processors available for a given lang & country
    """
    if debug:
        print(".. IMPORT FROM:", lang, "/", country, file=sys.stderr)
    if lang == TASK_ANY:
        name = TASK_ANY
        path = _LANG / TASK_ANY
    else:
        if country is None:
            country_elem = TASK_ANY
        elif country in ('in', 'is'):
            country_elem = country + '_'
        else:
            country_elem = country
        lang_elem = lang if lang not in ('is',) else lang + '_'
        name = f"{lang_elem}.{country_elem}"
        path = _LANG / lang_elem / country_elem

    # mod = importlib.import_module('...lang.' + name, __name__)
    return _gather_piitasks("pii_manager.lang." + name, path,
                            lang, country, debug=debug)


def _norm(elem: str) -> str:
    """
    Strip away underscores used to avoid reserved Python words
    """
    return elem[:-1] if elem.endswith('_') else elem


def country_list(lang: str) -> List[str]:
    """
    Return all countries for a given language
    """
    p = _LANG / lang
    return [_norm(d.name) for d in p.iterdir() if d.is_dir() and d.name != "__pycache__"]


def language_list() -> List[str]:
    return [_norm(d.name) for d in _LANG.iterdir() if d.is_dir() and d.name != "__pycache__"]


# --------------------------------------------------------------------------

_TASKS = None


def _gather_all_tasks(debug: bool = False):
    """
    Build the list of all tasks
    """
    global _TASKS

    if debug:
        print(".. DEFINED LANGUAGES:", " ".join(sorted(language_list())))

    _TASKS = {}
    for lang in language_list():
        if lang == TASK_ANY:
            _TASKS[lang] = import_processor(lang, debug=debug)
        else:
            _TASKS[lang] = {
                country: import_processor(lang, country, debug)
                for country in country_list(lang)
            }


def get_taskdict(debug: bool = False) -> Dict:
    """
    Return the dict holding all implemented pii tasks
    """
    global _TASKS
    if _TASKS is None:
        _gather_all_tasks(debug)
    return _TASKS

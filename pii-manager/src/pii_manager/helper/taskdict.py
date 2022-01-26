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
from collections import defaultdict
import re

from typing import Dict, List, Tuple, Callable, Any, Type, Union
from types import ModuleType

from pii_manager import PiiEnum
from .exception import InvArgException
from .base import BasePiiTask
from .types import TYPE_STR_LIST
from ..lang import LANG_ANY, COUNTRY_ANY

# Name of the list that holds the pii tasks at each module
_LISTNAME = "PII_TASKS"

# The structure holding all loaded tasks, as a language-keyed dictionary
_TASKS = None

# Locate the language folder
_LANG = Path(__file__).parents[1] / "lang"


# --------------------------------------------------------------------------


class InvPiiTask(InvArgException):
    def __init__(self, msg, lang=None, country=None):
        super().__init__(
            "task descriptor error [lang={}, country={}]: {}", lang, country, msg
        )


def _is_pii_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BasePiiTask)


def _import_task_object(objname: str) -> Union[Callable, Type[BasePiiTask]]:
    try:
        modname, oname = objname.rsplit(".", 1)
        mod = importlib.import_module(modname)
        return getattr(mod, oname)
    except Exception as e:
        raise InvPiiTask("cannot import task object '{}': {}", objname, e) from e


def _task_check(task: Dict, lang: str, country: TYPE_STR_LIST):
    """
    Check dict fields for a task, fill fields if needed
    """
    if not isinstance(task, dict):
        raise InvArgException("not a dictionary")
    if not isinstance(task.get("pii"), PiiEnum):
        raise InvArgException("field not a PiiEnum")

    # Check base fields: type & spec
    if "type" not in task:
        if _is_pii_class(task.get("task")):
            task["type"] = "PiiTask"
    if task.get("type") not in ("PiiTask", "callable", "re", "regex"):
        raise InvArgException("unsupported task type: {}", task.get("type"))
    if "task" not in task:
        raise InvArgException("invalid task specification: no task field")

    # Check task spec against task type
    if task["type"] in ("re", "regex") and not isinstance(task["task"], str):
        raise InvArgException("regex spec should be a string")
    elif task["type"] == "callable":
        if isinstance(task["task"], str):
            task["task"] = _import_task_object(task["task"])
        if not isinstance(task["task"], Callable):
            raise InvArgException("callable spec should be a callable")
    elif task["type"] == "PiiTask":
        if isinstance(task["task"], str):
            task["task"] = _import_task_object(task["task"])
        if not _is_pii_class(task["task"]):
            raise InvArgException("class spec should be a PiiTask object")

    # Fill in name
    if "name" not in task:
        name = getattr(task["task"], "pii_name", None)
        if not name:
            name = getattr(task["task"], "__name__", None)
            if name and task["type"] == "PiiTask":
                name = " ".join(re.findall(r"[A-Z][^A-Z]*", name)).lower()
            elif task["type"] == "callable":
                name = name.replace("_", " ")
        if not name:
            name = (task["type"] + " for " + task["pii"].name).lower()
        task["name"] = name

    # Fill in doc
    if "doc" not in task and not isinstance(task["task"], str):
        doc = getattr(task["task"], "__doc__", None)
        if doc:
            task["doc"] = doc.strip()

    # Process lang
    task_lang = task.get("lang")
    if (
        task_lang != lang
        and task_lang not in (None, LANG_ANY)
        and lang not in (None, LANG_ANY)
    ):
        raise InvArgException(
            "language mismatch in task descriptor: {} vs {}", task_lang, lang
        )
    elif task_lang is None:
        if lang is None:
            raise InvArgException("no lang can be determined")
        task["lang"] = lang

    # Process country
    if country is None:
        country = [COUNTRY_ANY]
    elif isinstance(country, str):
        country = [country]
    task_country = task.get("country")
    if (
        task_country not in country
        and task_country not in (None, COUNTRY_ANY)
        and COUNTRY_ANY not in country
    ):
        raise InvArgException(
            "country mismatch in task descriptor: {} vs {}", task_country, country
        )
    if task_country is None:
        task["country"] = country[0]
    if task["country"] == COUNTRY_ANY:
        task["country"] = None


def task_check(task: Dict, lang: str, country: str):
    """
    Check the fields in a task descriptor. Complete missing fields, if possible
    """
    try:
        _task_check(task, lang, country)
    except Exception as e:
        raise InvPiiTask(e, lang=lang, country=country)


def build_subdict(task_list: List[Tuple], lang: str, country: str = None) -> Dict:
    """
    Given a list of task tuples, build the task dict for them
    """
    if not isinstance(task_list, (list, tuple)):
        raise InvPiiTask("invalid tasklist: not a list/tuple", lang, country)

    subdict = defaultdict(list)
    for src in task_list:
        # Fetch the task
        if isinstance(src, tuple):  # parse a simplified form (tuple)
            # Checks
            if len(src) != 2 and (len(src) != 3 or not isinstance(src[1], str)):
                raise InvPiiTask("invalid simplified task spec", lang, country)
            # Task type
            task_type = (
                "PiiTask"
                if _is_pii_class(src[1])
                else "callable"
                if callable(src[1])
                else "regex"
                if isinstance(src[1], str)
                else None
            )
            # Build the dict
            td = {"pii": src[0], "type": task_type, "task": src[1]}
            if len(src) > 2:
                td["doc"] = src[2]
            task = td
        elif isinstance(src, dict):  # full form
            task = src.copy()
        else:
            raise InvPiiTask("element must be a tuple or dict", lang, country)

        # Check dict fields
        task_check(task, lang, country)
        # Add to dict
        subdict[task["pii"].name].append(task)

    return subdict


def _gather_piitasks(
    pkg: ModuleType, path: str, lang: str, country: str, debug: bool = False
) -> List[Tuple]:
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
    pii_tasks = defaultdict(list)
    for mname in modlist:
        mod = importlib.import_module("." + mname, pkg)
        task_list = getattr(mod, _LISTNAME, None)
        if task_list:
            subdict = build_subdict(task_list, lang, country)
            for name, value in subdict.items():
                pii_tasks[name] += value

    # If debug mode is on, print out the list
    if debug:
        if not pii_tasks:
            print("... NO PII TASKS for", pkg, file=sys.stderr)
        else:
            print("... PII TASKS for", pkg, file=sys.stderr)
            print("... path =", path, file=sys.stderr)
            for task_name, tasklist in pii_tasks.items():
                for task in tasklist:
                    print(
                        "  ",
                        task_name,
                        f"-> ({task['type']})",
                        task["doc"],
                        file=sys.stderr,
                    )

    return pii_tasks


def import_processor(lang: str, country: str = None, debug: bool = False) -> Dict:
    """
    Import all task processors available for a given lang & country
    """
    if debug:
        print(".. IMPORT FROM:", lang, "/", country, file=sys.stderr)
    if lang == LANG_ANY:
        name = LANG_ANY
        path = _LANG / LANG_ANY
    else:
        if country is None:
            country_elem = COUNTRY_ANY
        elif country in ("in", "is"):
            country_elem = country + "_"
        else:
            country_elem = country
        lang_elem = lang if lang not in ("is",) else lang + "_"
        name = f"{lang_elem}.{country_elem}"
        path = _LANG / lang_elem / country_elem

    # mod = importlib.import_module('...lang.' + name, __name__)
    return _gather_piitasks(
        "pii_manager.lang." + name, path, lang, country, debug=debug
    )


def _norm(elem: str) -> str:
    """
    Strip away underscores used to avoid reserved Python words
    """
    return elem[:-1] if elem.endswith("_") else elem


def country_list(lang: str) -> List[str]:
    """
    Return all countries for a given language
    """
    p = _LANG / lang
    return [
        _norm(d.name) for d in p.iterdir() if d.is_dir() and d.name != "__pycache__"
    ]


def language_list() -> List[str]:
    return [
        _norm(d.name) for d in _LANG.iterdir() if d.is_dir() and d.name != "__pycache__"
    ]


# --------------------------------------------------------------------------


def _gather_all_tasks(debug: bool = False):
    """
    Build the list of all tasks
    """
    global _TASKS

    if debug:
        print(". DEFINED LANGUAGES:", " ".join(sorted(language_list())))

    _TASKS = {}
    for lang in language_list():
        if lang == LANG_ANY:
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

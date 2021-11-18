'''
Traverse all folders and gather all implemented PiiTasks into a dictionary
'''

import sys
import importlib
from pathlib import Path

from typing import Dict, List, Tuple
from types import ModuleType

# Folder for language-independent tasks
TASK_ANY = 'any'

# Name of the list that holds the pii tasks at each module
_LISTNAME = 'PII_TASKS'

# --------------------------------------------------------------------------


_LANG = Path(__file__).parents[1] / 'lang'


def gather_piitasks(pkg: ModuleType, path: str,
                    debug: bool = False) -> List[Tuple]:
    antasks = {}
    modlist = (m.stem for m in Path(path).iterdir()
               if m.suffix == '.py' and m.stem != '__init__')
    for mname in modlist:
        mod = importlib.import_module('.' + mname, pkg)
        tasks = getattr(mod, _LISTNAME, None)
        if tasks:
            antasks.update({t[0].name: t for t in tasks})
    if debug:
        print(".. PII TASKS for", pkg, file=sys.stderr)
        for task_name, task in antasks.items():
            print("  ", task_name, '->', task[1], file=sys.stderr)
        print(".. path =", path, file=sys.stderr)

    return antasks


def import_processor(lang: str, country: str = None,
                     debug: bool = False) -> Dict:
    '''
    Import all task processors available for a given lang & country
    '''
    if debug:
        print('.. IMPORT FROM:', lang, country, file=sys.stderr)
    if lang == TASK_ANY:
        name = TASK_ANY
        path = _LANG / TASK_ANY
    elif country is None:
        name = f'{lang}.{TASK_ANY}'
        path = _LANG / lang / TASK_ANY
    else:
        name = f'{lang}.{country}'
        path = _LANG / lang / country

    #mod = importlib.import_module('...lang.' + name, __name__)
    return gather_piitasks('pii_manager.lang.' + name, path, debug=debug)


def country_list(lang: str) -> List[str]:
    '''
    Return all countries for a given language
    '''
    p = _LANG / lang
    return [d.name for d in p.iterdir()
            if d.is_dir() and d.name != '__pycache__']


def language_list() -> List[str]:
    return [d.name for d in _LANG.iterdir()
            if d.is_dir() and d.name != '__pycache__']


# --------------------------------------------------------------------------

_TASKS = None


def _gather_all_tasks(debug: bool = False):
    '''
    Build the list of all tasks
    '''
    global _TASKS

    if debug:
        print(".. LANGUAGES:", ','.join(sorted(language_list())))

    _TASKS = {}
    for lang in language_list():
        if lang == TASK_ANY:
            _TASKS[lang] = import_processor(lang, debug=debug)
        else:
            _TASKS[lang] = {country: import_processor(lang, country, debug)
                            for country in country_list(lang)}


def get_taskdict(debug: bool = False) -> Dict:
    '''
    Return the dicit holding all implemented pii tasks
    '''
    global _TASKS
    if _TASKS is None:
        _gather_all_tasks(debug)
    return _TASKS

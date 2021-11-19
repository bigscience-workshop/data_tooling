'''
Definition of the main PiiManager object
'''

from collections import defaultdict
from itertools import chain


from typing import Iterable, Tuple, List, Callable, Union, Dict

from ..piientity import PiiEntity
from ..piienum import PiiEnum
from ..helper import get_taskdict, TASK_ANY, country_list
from ..helper.base import BasePiiTask, CallablePiiTask, RegexPiiTask
from ..helper.exception import InvArgException


DEFAULT_TEMPLATES = {
    'replace': '<{name}>',
    'tag': '<{name}:{value}>'
}


# --------------------------------------------------------------------------


def fetch_all_tasks(lang: str, country: Iterable[str] = None,
                    debug: bool = False) -> Iterable[Tuple]:
    '''
    Return all available anonymizer tasks for a given language & (optionally)
    country
    '''
    taskdict = get_taskdict(debug=debug)
    # Language-independent
    for task in taskdict[TASK_ANY].values():
        yield task
    # Country-independent
    langdict = taskdict.get(lang, {})
    for task in langdict.get(TASK_ANY, {}).values():
        yield task
    # Country-specific
    if country:
        if country[0] == 'all':
            country = country_list(lang)
        for c in country:
            for task in langdict.get(c, {}).values():
                yield task


def fetch_task(taskname: str, lang: str,
               country: Iterable[str] = None) -> Iterable[Tuple]:
    '''
    Return a specific task for a given language & country
    (find the most specific task available)
    '''
    found = 0
    taskdict = get_taskdict()
    if isinstance(taskname, PiiEnum):
        taskname = taskname.name

    langdict = taskdict.get(lang, {})
    if langdict:
        # First try: language & country
        if country:
            if country[0] == 'all':
                country = country_list(lang)
            for c in country:
                task = langdict.get(c, {}).get(taskname)
                if task:
                    found += 1
                    yield task
        # Second try: only language
        task = langdict.get(TASK_ANY, {}).get(taskname)
        if task:
            found += 1
            yield task
    # Third try: generic task
    task = taskdict[TASK_ANY].get(taskname)
    if task:
        found += 1
        yield task

    # We didn't find anything
    if not found:
        print(f'Warning: cannot find any pii task for {taskname}, {lang}, {country}')


# --------------------------------------------------------------------------


def build_task(task, lang: str = None, country: str = None, ) -> BasePiiTask:
    if len(task) < 2:
        InvArgException('invalid task object: {}', task)
    pii = task[0]
    obj = task[1]
    if isinstance(obj, type(BasePiiTask)):
        proc = obj(pii=pii, lang=lang, country=country)
    elif isinstance(obj, Callable):
        proc = CallablePiiTask(obj, pii=pii, lang=lang, country=country)
    elif isinstance(obj, str):
        proc = RegexPiiTask(obj, task[2] if len(task) > 2 else pii.name,
                            pii=pii, lang=lang, country=country)
    else:
        raise InvArgException('invalid pii task object for {}: {}', pii.name,
                              type(obj))
    return proc


# --------------------------------------------------------------------------

class PiiManager:

    def __init__(self, lang: str, country: List[str] = None,
                 tasks: Iterable[PiiEnum] = None,
                 all_tasks: bool = False, mode: str = None,
                 template: str = None, debug: bool = False):
        '''
        Initalize an anonymizer object, loading & initializing all specified
        processing tasks
        '''
        # Sanitize input
        self.lang = lang.lower()
        if isinstance(country, str):
            country = [country]
        self.country = [c.lower() for c in country] if country else None
        self.mode = mode if mode is not None else 'replace'
        if template is None and self.mode != 'extract':
            template = DEFAULT_TEMPLATES[self.mode]
        self.template = template

        # Get the list of tasks we will use
        if all_tasks:
            tasklist = fetch_all_tasks(self.lang, self.country, debug=debug)
        else:
            if isinstance(tasks, PiiEnum):
                tasks = [tasks]
            tasklist = (fetch_task(name, self.lang, self.country)
                        for name in tasks)
            tasklist = filter(None, chain.from_iterable(tasklist))

        # Build an ordered array of tasks processors
        taskproc = (build_task(t, mode, template) for t in tasklist)
        self.tasks = sorted(taskproc, key=lambda e: e.pii.value)
        self.stats = defaultdict(int)


    def __call__(self, doc: str) -> Union[str, Iterable[PiiEntity]]:
        '''
        Process a document, calling all defined anonymizers
        '''
        if self.mode == 'extract':
            return self.mode_extract(doc)
        else:
            return self.mode_subst(doc)


    def mode_subst(self, doc: str) -> str:
        '''
        Process a document, calling all defined processors and performing
        PII substitution
        '''
        self.stats['calls'] += 1
        for task_proc in self.tasks:
            output = []
            pos = 0
            for pii in task_proc(doc):
                output += [doc[pos:pii.pos],
                           self.template.format(name=pii.elem.name,
                                                value=pii.value)]
                self.stats[pii.elem.name] += 1
                pos = pii.pos + len(pii)
            doc = ''.join(output) + doc[pos:]
        return doc


    def mode_extract(self, doc: str) -> Iterable[PiiEntity]:
        '''
        Process a document, calling all defined processors and performing
        PII extraction
        '''
        self.stats['calls'] += 1
        for task_proc in self.tasks:
            elem_list = task_proc(doc)
            for pii in elem_list:
                yield pii
                self.stats[pii.elem.name] += 1


    def task_info(self) -> Dict:
        '''
        Return a dictionary with all defined tasks
        '''
        return {task.pii: (task.doc or task.__doc__).strip()
                for task in self.tasks}

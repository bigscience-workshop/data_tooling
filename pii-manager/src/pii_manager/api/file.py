"""
File-based API
"""

import sys
import re
import json
import gzip
import bz2
import lzma
from itertools import zip_longest
from pathlib import Path

from typing import Dict, List, TextIO, Iterable, Optional, Union

from pii_manager import PiiEnum
from pii_manager.api import PiiManager
from pii_manager.piientity import PiiEntity, piientity_asdict
from pii_manager.helper.exception import PiiManagerException, InvArgException
from pii_manager.helper.json import CustomJSONEncoder
from pii_manager.helper.types import TYPE_STR_LIST

TYPE_RESULT = Union[str, Iterable[PiiEntity]]


def openfile(name: str, mode: str) -> TextIO:
    """
    Open files, raw text or compressed (gzip, bzip2 or xz)
    """
    name = str(name)
    if name == "-":
        return sys.stdout if mode.startswith("w") else sys.stdin
    elif name.endswith(".gz"):
        return gzip.open(name, mode, encoding="utf-8")
    elif name.endswith(".bz2"):
        return bz2.open(name, mode, encoding="utf-8")
    elif name.endswith(".xz"):
        return lzma.open(name, mode, encoding="utf-8")
    else:
        return open(name, mode, encoding="utf-8")


def sentence_splitter(doc: str) -> Iterable[str]:
    """
    Split text by sentence separators
     (keeping the separator at the end of the sentence, so that joining the
    pieces recovers exactly the same text)
    """
    split = re.split(r"(\s*[\.!\?．。]\s+)", doc)
    args = [iter(split)] * 2
    for sentence, sep in zip_longest(*args, fillvalue=""):
        if sentence:
            yield sentence + sep


def write_extract(result: Iterable[PiiEntity], index: Dict, out: TextIO):
    """
    Write output for "extract" mode as NDJSON
      :param result: iterable of result PiiEntity objects
      :param index: indexing information to be added to each PII_TASKS
      :param out: destination to write the NDJSON lines to
    """
    for pii in result:
        elem = piientity_asdict(pii)
        if index:
            elem.update(index)
        json.dump(elem, out, ensure_ascii=False)
        print(file=out)


def write(result: TYPE_RESULT, mode: str, index: Optional[Dict], out: TextIO):
    """
    Write processing result to output
    """
    if mode == "extract":
        write_extract(result, index, out)
    elif mode == "full":
        json.dump(result, out, ensure_ascii=False, cls=CustomJSONEncoder)
    else:
        out.write(result)


def print_tasks(proc: PiiManager, out: TextIO):
    print("\n. Installed tasks:", file=out)
    for (pii, country), doc in proc.task_info().items():
        print(f" {pii.name}  [country={country}]\n   ", doc, file=out)


def read_taskfile(filename: str) -> List[Dict]:
    """
    Read a list of task descriptors from a JSON file
    """
    with open(filename, encoding="utf-8") as f:
        try:
            tasklist = json.load(f)
            for td in tasklist:
                td["pii"] = PiiEnum[td["pii"]]
            return tasklist
        except json.JSONDecodeError as e:
            raise InvArgException("invalid task spec file {}: {}", filename, e)
        except KeyError as e:
            if str(e) == "pii":
                raise InvArgException(
                    "missing 'pii' field in task descriptor in {}", filename
                )
            else:
                raise InvArgException(
                    "cannot find PiiEnum element '{}' for task descriptor in {}",
                    e,
                    filename,
                )
        except Exception as e:
            raise InvArgException("cannot read taskfile '{}': {}", filename, e)


def add_taskfile(filename: TYPE_STR_LIST, proc: PiiManager):
    """
    Add all tasks defined in a JSON file (or several) to a processing object
    """
    if isinstance(filename, (str, Path)):
        filename = [filename]
    for name in filename:
        tasklist = read_taskfile(name)
        proc.add_tasks(tasklist)


# ----------------------------------------------------------------------


def process_file(
    infile: str,
    outfile: str,
    lang: str,
    country: List[str] = None,
    tasks: List[str] = None,
    all_tasks: bool = False,
    taskfile: TYPE_STR_LIST = None,
    split: str = "line",
    mode: str = "replace",
    template: str = None,
    debug: bool = False,
    show_tasks: bool = False,
    show_stats: bool = False,
) -> Dict:
    """
    Process a number of PII tasks on a text file
    """
    # Create the object
    proc = PiiManager(
        lang,
        country,
        tasks,
        all_tasks=all_tasks,
        mode=mode,
        template=template,
        debug=debug,
    )
    if taskfile:
        add_taskfile(taskfile, proc)
    if show_tasks:
        print_tasks(proc, sys.stderr)

    # Process the file
    print(". Reading from:", infile, file=sys.stderr)
    print(". Writing to:", outfile, file=sys.stderr)
    with openfile(infile, "rt") as fin:
        with openfile(outfile, "wt") as fout:
            if split == "block":
                write(proc(fin.read()), mode, None, fout)
            elif split == "line":
                for n, line in enumerate(fin):
                    write(proc(line), mode, {"line": n + 1}, fout)
            elif split == "sentence":
                for n, sentence in enumerate(sentence_splitter(fin.read())):
                    write(proc(sentence), mode, {"sentence": n + 1}, fout)
            else:
                raise PiiManagerException("invalid split mode: {}", split)

    if show_stats:
        print("\n. Statistics:", file=sys.stderr)
        for k, v in proc.stats.items():
            print(f"  {k:20} :  {v:5}", file=sys.stderr)

    return proc.stats

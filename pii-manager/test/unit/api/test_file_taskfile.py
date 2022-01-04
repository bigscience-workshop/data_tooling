from pathlib import Path
import tempfile

import pytest

from pii_manager import PiiEnum
from pii_manager.api import process_file, PiiManager
from pii_manager.api.file import read_taskfile, add_taskfile
from pii_manager.helper.exception import InvArgException


def datafile(name: str) -> str:
    return Path(__file__).parents[2] / "data" / name


def readfile(name: str) -> str:
    with open(name, "rt", encoding="utf-8") as f:
        return f.read().strip()


def test10_taskfile():
    """
    Read a taskfile
    """
    taskfile = datafile("taskfile.json")
    tasklist = read_taskfile(taskfile)
    assert len(tasklist) == 3
    assert tasklist[0]["pii"] == PiiEnum.IP_ADDRESS
    assert tasklist[1]["pii"] == PiiEnum.BITCOIN_ADDRESS
    assert tasklist[2]["pii"] == PiiEnum.CREDIT_CARD


def test11_taskfile_error():
    """
    Read a taskfile with an error
    """
    taskfile = datafile("taskfile-error.json")
    with pytest.raises(InvArgException):
        read_taskfile(taskfile)


def test12_taskfile():
    """
    Read a taskfile
    """
    taskfile = datafile("taskfile.json")
    proc = PiiManager("en")
    add_taskfile(taskfile, proc)
    got = proc.task_info()
    exp = {
        (PiiEnum.CREDIT_CARD, None): [("credit card", "credit card number detection")],
        (PiiEnum.BITCOIN_ADDRESS, None): [
            ("bitcoin address", "bitcoin address detection")
        ],
        (PiiEnum.IP_ADDRESS, None): [
            ("regex for ip_address", "ip address detection via regex")
        ],
    }
    assert exp == got


def test13_taskfile_err():
    """
    Read a taskfile, try to add to an object with a language mismatch
    """
    taskfile = datafile("taskfile.json")
    proc = PiiManager("fr")
    with pytest.raises(InvArgException):
        add_taskfile(taskfile, proc)


def test20_taskfile():
    """
    Read a taskfile, process data
    """
    with tempfile.NamedTemporaryFile() as f:
        stats = process_file(
            datafile("orig.txt"), f.name, "en", taskfile=datafile("taskfile.json")
        )
        exp = {"calls": 3, "CREDIT_CARD": 1, "BITCOIN_ADDRESS": 1}
        assert stats == exp

        exp = readfile(datafile("replace.txt"))
        got = readfile(f.name)
        # print(got)
        assert got == exp

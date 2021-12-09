from pathlib import Path
import tempfile

import pytest

from pii_manager import PiiEnum
from pii_manager.api import process_file


def datafile(name: str) -> str:
    return Path(__file__).parents[2] / "data" / name


def readfile(name: str) -> str:
    with open(name, "rt", encoding="utf-8") as f:
        return f.read().strip()


@pytest.mark.parametrize("mode", ["replace", "tag", "extract", "full"])
def test10_line(mode):
    """
    Test splitting the file by lines
    """
    with tempfile.NamedTemporaryFile() as f:
        tasks = [PiiEnum.CREDIT_CARD, PiiEnum.BITCOIN_ADDRESS]
        stats = process_file(datafile("orig.txt"), f.name, "en", tasks=tasks, mode=mode)
        exp = {"calls": 3, "CREDIT_CARD": 1, "BITCOIN_ADDRESS": 1}
        assert stats == exp

        name = mode + (".txt" if mode in ("replace", "tag") else "-line.ndjson")
        exp = readfile(datafile(name))
        got = readfile(f.name)
        # print(got)
        assert got == exp


@pytest.mark.parametrize("mode", ["replace", "tag", "extract", "full"])
def test20_block(mode):
    """
    Test whole files
    """
    with tempfile.NamedTemporaryFile() as f:
        tasks = [PiiEnum.CREDIT_CARD, PiiEnum.BITCOIN_ADDRESS]
        stats = process_file(
            datafile("orig.txt"), f.name, "en", tasks=tasks, mode=mode, split="block"
        )
        exp = {"calls": 1, "CREDIT_CARD": 1, "BITCOIN_ADDRESS": 1}
        assert stats == exp

        name = mode + (".txt" if mode in ("replace", "tag") else "-block.ndjson")
        exp = readfile(datafile(name))
        got = readfile(f.name)
        assert got == exp


@pytest.mark.parametrize("mode", ["replace", "tag", "extract", "full"])
def test30_sentence(mode):
    """
    Test splitting the file by sentences
    """
    with tempfile.NamedTemporaryFile() as f:
        tasks = [PiiEnum.CREDIT_CARD, PiiEnum.BITCOIN_ADDRESS]
        stats = process_file(
            datafile("orig.txt"), f.name, "en", tasks=tasks, mode=mode, split="sentence"
        )
        exp = {"calls": 2, "CREDIT_CARD": 1, "BITCOIN_ADDRESS": 1}
        assert stats == exp

        name = mode + (".txt" if mode in ("replace", "tag") else "-sentence.ndjson")
        exp = readfile(datafile(name))
        got = readfile(f.name)
        assert got == exp

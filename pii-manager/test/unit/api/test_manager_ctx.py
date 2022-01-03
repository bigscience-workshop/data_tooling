"""
Test base objects with context
"""

from pii_manager import PiiEnum, PiiEntity
from pii_manager.api import PiiManager


def _pii(pos):
    return PiiEntity(PiiEnum.GOV_ID, pos, "3451-K", country="vo", name="vogonian ID")


TEST = [
    ("my Vogon ID is 3451-K", [_pii(15)]),
    ("the number 3451-K is my Vogonian ID", [_pii(11)]),
    ("the Vogon ID are 3451-K", []),  # context outside window
    ("my Betelgeuse ID is 3451-K", []),  # context does not match
]


# ------------------------------------------------------------------------

DUMMY_REGEX = {
    "pii": PiiEnum.GOV_ID,
    "type": "regex",
    "task": r"""\b\d{4}-\w\b""",
    "lang": "en",
    "name": "vogonian ID",
    "country": "vo",
    "doc": "a toy example to match a government id",
    "context": {"value": ["Vogon ID", "vogonian id"], "width": [12, 20]},
}


def test10_context_regex():
    """
    Check a PII task with contexts, regex variant
    """
    obj = PiiManager("en", mode="extract")
    obj.add_tasks([DUMMY_REGEX])
    for (text, exp) in TEST:
        got = obj(text)
        assert list(got) == exp


# ------------------------------------------------------------------------


DUMMY_CLASS = {
    "pii": PiiEnum.GOV_ID,
    "type": "PiiTask",
    "task": "unit.api.test_manager_add.DummyPii",
    "lang": "en",
    "country": "vo",
    "name": "vogonian ID",
    "doc": "a toy example to match a government id",
    "context": {"value": ["Vogon ID", "vogonian id"], "width": [12, 20]},
}


def test20_context_class():
    """
    Check a PII task with contexts, class variant
    """
    obj = PiiManager("en", mode="extract")
    obj.add_tasks([DUMMY_CLASS])
    for (text, exp) in TEST:
        got = obj(text)
        assert list(got) == exp

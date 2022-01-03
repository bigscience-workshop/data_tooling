"""
Test adding an external Pii task
"""

import re

from pii_manager import PiiEnum, PiiEntity
from pii_manager.api import PiiManager
from pii_manager.lang import COUNTRY_ANY
from pii_manager.helper.base import BasePiiTask


# ---------------------------------------------------------------------

DUMMY_REGEX = {
    "pii": PiiEnum.DISEASE,
    "type": "regex",
    "task": r"""\b(cancer|leuka?emia|aids)\b""",
    "lang": "en",
    "country": COUNTRY_ANY,
    "name": "disease names",
    "doc": "a toy example to match some disease names",
}


TEST_REGEX = [
    (
        "Alvin had cancer detected and his email is alvin@anywhere.com",
        "Alvin had <DISEASE> detected and his email is <EMAIL_ADDRESS>",
    )
]


def test100_info():
    obj = PiiManager("en", None, PiiEnum.EMAIL_ADDRESS)
    obj.add_tasks([DUMMY_REGEX])
    exp = {
        (PiiEnum.EMAIL_ADDRESS, None): [("regex for email_address", "Email address")],
        (PiiEnum.DISEASE, None): [
            ("disease names", "a toy example to match some disease names")
        ],
    }
    assert obj.task_info() == exp


def test110_call():
    obj = PiiManager("en", None, PiiEnum.EMAIL_ADDRESS)
    obj.add_tasks([DUMMY_REGEX])

    for (doc, exp) in TEST_REGEX:
        got = obj(doc)
        assert got == exp


# ---------------------------------------------------------------------


class DummyPii(BasePiiTask):
    def find(self, doc):
        for r in re.finditer(r"\d{4}-\w", doc):
            yield PiiEntity(
                PiiEnum.GOV_ID,
                r.start(),
                r.group(),
                country=self.country,
                name=self.name,
            )


DUMMY_CLASS = {
    "pii": PiiEnum.GOV_ID,
    "type": "PiiTask",
    "task": "unit.api.test_manager_add.DummyPii",
    "lang": "en",
    "country": "vo",
    "doc": "a toy example to match some disease names",
}


TEST_CLASS = [
    ("Jeltz has vogonian ID number 1234-J", "Jeltz has vogonian ID number <GOV_ID>")
]


def test200_call():
    obj = PiiManager("en")
    obj.add_tasks([DUMMY_CLASS])

    for (doc, exp) in TEST_CLASS:
        got = obj(doc)
        assert got == exp

"""
Test US Social Security Number
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager

TEST = [
    # A valid SSN
    ("SSN: 536-90-4399", "SSN: <GOV_ID>"),
    # SSN with spaces
    ("SSN: 536 90 4399", "SSN: <GOV_ID>"),
    # An invalid SSN
    ("not a SSN: 666-90-4399", "not a SSN: 666-90-4399"),
]


def test10_ssn():
    obj = PiiManager("en", "US", PiiEnum.GOV_ID)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp

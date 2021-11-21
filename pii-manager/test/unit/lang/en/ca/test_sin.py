"""
Test Canadian Social Insurance Number
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager

TEST = [
    # A valid SIN
    ("SIN: 963-553-151", "SIN: <GOV_ID>"),
    # SIN with spaces
    ("SIN: 339 892 317 number", "SIN: <GOV_ID> number"),
    # An invalid SIN
    ("not a SIN: 123-456-781", "not a SIN: 123-456-781"),
]


def test10_ssn():
    obj = PiiManager("en", "CA", PiiEnum.GOV_ID)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp

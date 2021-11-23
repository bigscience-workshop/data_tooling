"""
Test Indian Aadhaar Number
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager

TEST = [
    # A valid aadhaar
    ("aadhaar number 234123412346", "aadhaar number <GOV_ID>"),
    # aadhaar with spaces
    ("aadhaar number 2341 2341 2346", "aadhaar number <GOV_ID>"),
    # An invalid aadhaar
    (
        "not a real aadhaar number: 2341 2341 2347",
        "not a real aadhaar number: 2341 2341 2347",
    ),
]


def test10_ssn():
    obj = PiiManager("en", "IN", PiiEnum.GOV_ID)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp

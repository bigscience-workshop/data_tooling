"""
Test Australian Business Number
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager

TEST = [
    # A valid ABN
    ("business number: 83 914 571 673.", "business number: <GOV_ID>."),
    # ABN without spaces
    ("business number: 83914571673.", "business number: <GOV_ID>."),
    # An invalid ABN
    ("not an ABN: 83 914 571 679", "not an ABN: 83 914 571 679"),
]


def test10_abn():
    obj = PiiManager("en", "AU", PiiEnum.GOV_ID)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp

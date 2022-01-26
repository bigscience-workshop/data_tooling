"""
Test Australian Tax File Number
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager

TEST = [
    # A valid ABN
    ("tax file number: 963 553 151.", "tax file number: <GOV_ID>."),
    ("the tfn is: 123 456 782", "the tfn is: <GOV_ID>"),
    # TFN without spaces
    ("tax file number: 963553151.", "tax file number: <GOV_ID>."),
    # An invalid TFN
    ("not a TFN: 123 456 781", "not a TFN: 123 456 781"),
]


def test10_abn():
    obj = PiiManager("en", "AU", PiiEnum.GOV_ID)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp

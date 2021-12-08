"""
Test Chinese government ids (Resident Identity Card & Passport)
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager

TEST = [
    # A valid RIC
    ("公民身份号码 360426199101010071", "公民身份号码 <GOV_ID>"),
    # An invalid RIC
    ("公民身份号码 360426199101010072", "公民身份号码 360426199101010072"),
    # An invalid RIC (one aditional digit)
    ("公民身份号码 3604261991010100717", "公民身份号码 3604261991010100717"),
    # A correct passport number
    ("中华人民共和国护照 D12345678", "中华人民共和国护照 <GOV_ID>"),
    # An incorrect passport number (invalid letter)
    ("中华人民共和国护照 K12345678", "中华人民共和国护照 K12345678"),
    # An incorrect passport number (only 7 digits)
    ("中华人民共和国护照 D1234567", "中华人民共和国护照 D1234567"),
]


def test10_ssn():
    obj = PiiManager("zh", "CN", PiiEnum.GOV_ID)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp

"""
Test international phone numbers
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager
from pii_manager.lang import LANG_ANY

TEST = [
    # Standard phone number
    ("phone number: +34 983 453 999", "phone number: <PHONE_NUMBER>"),
    ("phone number: +34983453999", "phone number: <PHONE_NUMBER>"),
    ("ph. +34983453999", "ph. <PHONE_NUMBER>"),
    # An invalid country code
    ("phone number: +99 983 453 999", "phone number: +99 983 453 999"),
    # No valid contexts
    ("number: +34983453999", "number: +34983453999"),
    ("phonograph +34983453999", "phonograph +34983453999"),
]


def test10_ssn():
    obj = PiiManager("en", LANG_ANY, PiiEnum.PHONE_NUMBER)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp

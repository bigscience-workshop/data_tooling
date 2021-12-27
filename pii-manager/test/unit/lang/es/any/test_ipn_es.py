"""
Test international phone numbers
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager
from pii_manager.lang import LANG_ANY

TEST = [
    # Standard phone number
    ("teléfono: +34 983 453 999", "teléfono: <PHONE_NUMBER>"),
    ("tf. +34983453999", "tf. <PHONE_NUMBER>"),
    ("numero de telefono +34983453999", "numero de telefono <PHONE_NUMBER>"),
    # An invalid country code
    ("teléfono: +99 983 453 999", "teléfono: +99 983 453 999"),
    # No valid contexts
    ("número: +34983453999", "número: +34983453999"),
    ("tff +34983453999", "tff +34983453999"),
]


def test10_ssn():
    obj = PiiManager("es", LANG_ANY, PiiEnum.PHONE_NUMBER)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp

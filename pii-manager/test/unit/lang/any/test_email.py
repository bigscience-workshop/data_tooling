"""
Test email addersses
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager


TEST = [
    # A valid email address
    (
        "My email is anyone@whatever.com.",
        "My email is <EMAIL_ADDRESS>.",
    ),
    # An invalid email address
    (
        "My email is anyone@whatever.",
        "My email is anyone@whatever.",
    ),
]


def test10_credit_card():
    obj = PiiManager("es", None, PiiEnum.EMAIL_ADDRESS)
    for doc, exp in TEST:
        got = obj(doc)
        assert exp == got

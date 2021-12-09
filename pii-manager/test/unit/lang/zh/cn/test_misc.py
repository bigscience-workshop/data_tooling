"""
Test PII elements for Chinese (Phone numbers, street addresses & diseases)
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager

TEST = [
    # Phone number
    ("045-4123456", "<PHONE_NUMBER>"),
    # Not a phone number (too many digits in the first part)
    ("70045-4123456", "70045-4123456"),
    # ----- We are missing here tests for STREET_ADDRESS & DISEASE
]


def test10_ssn():
    obj = PiiManager(
        "zh", "CN", [PiiEnum.STREET_ADDRESS, PiiEnum.PHONE_NUMBER, PiiEnum.DISEASE]
    )
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp

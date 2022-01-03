"""
Test IP addresses
"""

from pii_manager import PiiEnum
from pii_manager.api import PiiManager


TEST = [
    # A valid IP address
    (
        "My IP address is 10.45.122.65",
        "My IP address is <IP_ADDRESS>",
    ),
    # An invalid IP address
    ("My IP address is 310.45.122.65", "My IP address is 310.45.122.65"),
    # An IP address without context
    ("My address is 10.45.122.65", "My address is 10.45.122.65"),
]


def test10_ip_address():
    obj = PiiManager("en", None, PiiEnum.IP_ADDRESS)
    for doc, exp in TEST:
        got = obj(doc)
        assert exp == got

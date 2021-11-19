"""
Test bitcoin addresses
"""


from pii_manager import PiiEnum
from pii_manager.api import PiiManager


TEST = [
    # A valid bitcoin address
    (
        "BTC address: 1JayVxfVgdaFKirkZTZVK4CdRnFDdFNENN",
        "BTC address: <BITCOIN_ADDRESS>",
    ),
    (
        "BTC address: bc1qwxxvjxlakxe9rmxcphh4yy8a2t6z00k4gc4mpj",
        "BTC address: <BITCOIN_ADDRESS>",
    ),
    # An invalid bitcoin address
    (
        "BTC address: 1AGNa15ZQXAZUgFiqJ2i7Z2DPU2J6hW623",
        "BTC address: 1AGNa15ZQXAZUgFiqJ2i7Z2DPU2J6hW623",
    ),
]


def test10_credit_card():
    obj = PiiManager("en", None, PiiEnum.BITCOIN_ADDRESS)
    for doc, exp in TEST:
        got = obj(doc)
        assert got == exp
